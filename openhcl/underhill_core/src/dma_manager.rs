// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! This module provides a global DMA manager and client implementation.
//! It manages DMA buffers and provides clients with access to these buffers.
//! The `GlobalDmaManager` creates DMA buffers for different devices.
//! The `DmaClientImpl` struct implements the `user_driver::DmaClient` trait.

use guestmem::ranges::PagedRange;
use guestmem::GuestMemory;
use memory_range::MemoryRange;
use parking_lot::Condvar;
use parking_lot::Mutex;
use std::sync::Arc;
use user_driver::memory::MemoryBlock;
use user_driver::memory::PAGE_SIZE;
use user_driver::memory::PAGE_SIZE32;
use user_driver::memory::PAGE_SIZE64;
use user_driver::vfio::VfioDmaBuffer;
use user_driver::DmaBuffer;
use user_driver::DmaError;
use user_driver::DmaTransaction;
use user_driver::MemoryBacking;
use virt_mshv_vtl::UhPartition;

pub struct GlobalDmaManager {
    inner: Arc<Mutex<GlobalDmaManagerInner>>,
}

pub struct GlobalDmaManagerInner {
    dma_buffer_spawner: Box<dyn Fn(String) -> anyhow::Result<Arc<dyn VfioDmaBuffer>> + Send>,
    partition: Arc<UhPartition>,
}

impl GlobalDmaManagerInner {
    pub fn manager_map_dma_transaction(&self, ranges: &[MemoryRange]) -> anyhow::Result<()> {
        self.partition
            .as_ref()
            .pin_gpa_ranges(ranges)
            .map_err(|e| anyhow::anyhow!("Failed to map DMA transaction: {:?}", e))?;
        Ok(())
    }

    pub fn manager_unmap_dma_transaction(&self, ranges: &[MemoryRange]) -> anyhow::Result<()> {
        self.partition
            .as_ref()
            .unpin_gpa_ranges(ranges)
            .map_err(|e| anyhow::anyhow!("Failed to unmap DMA transaction: {:?}", e))?;
        Ok(())
    }
}

impl GlobalDmaManager {
    /// Creates a new `GlobalDmaManager` with the given DMA buffer spawner.
    pub fn new(
        dma_buffer_spawner: Box<dyn Fn(String) -> anyhow::Result<Arc<dyn VfioDmaBuffer>> + Send>,
        partition: Arc<UhPartition>,
    ) -> Self {
        let inner = GlobalDmaManagerInner {
            dma_buffer_spawner,
            partition,
        };

        GlobalDmaManager {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    fn create_client_internal(
        inner: &Arc<Mutex<GlobalDmaManagerInner>>,
        device_name: String,
    ) -> anyhow::Result<Arc<DmaClientImpl>> {
        let manager_inner = inner.lock();
        let allocator = {
            // Access the page_pool and call its allocator method directly
            (manager_inner.dma_buffer_spawner)(device_name)
                .map_err(|e| anyhow::anyhow!("failed to get DMA buffer allocator: {:?}", e))?
        };

        let mem = allocator
            .create_dma_buffer((100 * PAGE_SIZE32).try_into().unwrap())
            .map_err(|e| anyhow::anyhow!("Failed to create DMA buffer: {:?}", e))?;

        let client = DmaClientImpl {
            _dma_manager_inner: inner.clone(),
            dma_buffer_allocator: Some(allocator.clone()),
            allocator: Mutex::new(BounceBufferAllocator::new(mem.len())),
            mem,
            condvar: Condvar::new(),
        };

        let arc_client = Arc::new(client);

        Ok(arc_client)
    }

    /// Returns a `DmaClientSpawner` for creating DMA clients.
    pub fn get_client_spawner(&self) -> DmaClientSpawner {
        DmaClientSpawner {
            dma_manager_inner: self.inner.clone(),
        }
    }
}

pub struct DmaClientImpl {
    /// This is added to support map/pin functionality in the future.
    _dma_manager_inner: Arc<Mutex<GlobalDmaManagerInner>>,
    dma_buffer_allocator: Option<Arc<dyn VfioDmaBuffer>>,
    allocator: Mutex<BounceBufferAllocator>,
    mem: MemoryBlock,
    condvar: Condvar,
}

impl DmaClientImpl {
    /// Blocking allocation: waits until memory is available.
    fn allocate(&self, size: usize) -> DmaBuffer {
        let mut alloc = self.allocator.lock();
        loop {
            if let Some((id, offset)) = alloc.malloc(size) {
                return DmaBuffer { id, offset, size };
            }
            // Wait for memory to be freed.
            self.condvar.wait(&mut alloc);
        }
    }

    /// Free a previously allocated DMA buffer and notify waiting threads.
    fn free(&self, buffer: DmaBuffer) {
        let mut alloc = self.allocator.lock();
        alloc.free(buffer.id);
        // Wake up one waiting thread.
        self.condvar.notify_one();
    }

    fn allocate_bounce_buffer(
        &self,
        ranges: &[MemoryRange],
        total_size: usize,
        options: user_driver::DmaTransectionOptions,
    ) -> Result<user_driver::DmaTransactionHandler, DmaError> {
        let bounce_buffer = self.allocate(total_size);
        let mut cumulative_offset = 0;

        let transactions: Vec<DmaTransaction> = ranges
            .iter()
            .map(|range| {
                let range_size = range.len();
                let transaction_offset = bounce_buffer.offset + cumulative_offset;
                cumulative_offset += range_size as usize;

                DmaTransaction::new(
                    DmaBuffer::new(0, transaction_offset, range_size as usize),
                    range.start(),
                    options.clone(),
                    MemoryBacking::BounceBuffer,
                )
            })
            .collect();

        Ok(user_driver::DmaTransactionHandler { transactions })
    }

    fn handle_dma_mapping_failure(
        &self,
        ranges: &[MemoryRange],
        guest_memory: &GuestMemory,
        mem: PagedRange<'_>,
        options: user_driver::DmaTransectionOptions,
    ) -> Result<user_driver::DmaTransactionHandler, DmaError> {
        let mut transactions = Vec::new();

        for range in ranges {
            let range_size = range.len() as usize;
            let bounce_buffer = self.allocate(range_size); // Allocate a separate buffer per range

            let mut remaining = range_size;
            let page_count = bounce_buffer.size / PAGE_SIZE64 as usize;

            for i in 0..page_count {
                let len = PAGE_SIZE32.min(remaining as u32) as usize;
                remaining -= len;

                let effective_offset = bounce_buffer.offset + i * PAGE_SIZE32 as usize;
                let offset_in_page = effective_offset % PAGE_SIZE32 as usize;
                let dest_page = effective_offset / PAGE_SIZE32 as usize;

                guest_memory
                    .read_range_to_atomic(
                        &mem.subrange(i * PAGE_SIZE32 as usize, len),
                        &self.mem.as_slice()[(dest_page * PAGE_SIZE + offset_in_page) as usize..][..len],
                    )
                    .map_err(|_| DmaError::BounceBufferFailed)?;
            }

            let dma_transaction = DmaTransaction::new(
                bounce_buffer,
                range.start(),
                options.clone(),
                MemoryBacking::BounceBuffer,
            );

            transactions.push(dma_transaction);
        }

        Ok(user_driver::DmaTransactionHandler { transactions })
    }
}

impl user_driver::DmaClient for DmaClientImpl {
    fn allocate_dma_buffer(&self, total_size: usize) -> anyhow::Result<MemoryBlock> {
        if self.dma_buffer_allocator.is_none() {
            return Err(anyhow::anyhow!("DMA buffer allocator is not set"));
        }

        let allocator = self.dma_buffer_allocator.as_ref().unwrap();

        allocator.create_dma_buffer(total_size)
    }

    fn attach_dma_buffer(&self, len: usize, base_pfn: u64) -> anyhow::Result<MemoryBlock> {
        let allocator = self.dma_buffer_allocator.as_ref().unwrap();
        allocator.restore_dma_buffer(len, base_pfn)
    }

    fn map_dma_ranges(
        &self,
        guest_memory: &GuestMemory,
        mem: PagedRange<'_>,
        options: user_driver::DmaTransectionOptions,
    ) -> Result<user_driver::DmaTransactionHandler, DmaError> {
        let ranges = mem.memoryranges();
        let transactions ;

        if options.is_rx {
            return self.allocate_bounce_buffer(&ranges, mem.len(), options);
        }

        match self
            ._dma_manager_inner
            .lock()
            .manager_map_dma_transaction(&ranges)
        {
            Ok(_) => {
                transactions = ranges
                    .iter()
                    .map(|range| {
                        DmaTransaction::new(
                            DmaBuffer::new(0, 0, range.len() as usize),
                            range.start(),
                            options.clone(),
                            MemoryBacking::Pinned,
                        )
                    })
                    .collect();
            }
            Err(_) => return self.handle_dma_mapping_failure(&ranges, guest_memory, mem, options),
        }

        Ok(user_driver::DmaTransactionHandler { transactions })
    }

    fn unmap_dma_ranges(
        &self,
        guest_memory: &GuestMemory,
        mem: PagedRange<'_>,
        dma_transactions: &[DmaTransaction],
    ) -> Result<(), DmaError> {
        let mut ranges: Vec<MemoryRange> = Vec::new();
        let mut bounce_buffers: Vec<DmaBuffer> = Vec::new(); // Track buffers for deallocation

        // Copy data back from the bounce buffer to user memory for RX transactions
        for transaction in dma_transactions {
            if transaction.options().is_rx && transaction.backing() == MemoryBacking::BounceBuffer {
                let src_offset = transaction.offset();
                let src_len = transaction.size() as usize;

                let dest_addr = transaction.original_addr();
                let dest_len = src_len;

                // Ensure that the source slice is within the allocated buffer range
                if src_offset + src_len > self.mem.as_slice().len() {
                    return Err(DmaError::UnmapFailed);
                }

                let src_slice = &self.mem.as_slice()[src_offset..src_offset + src_len];
                // Convert the effective offset into a destination page and an offset within that page.
                let dest_page = dest_addr / PAGE_SIZE64;
                let offset_in_page = dest_addr % PAGE_SIZE64;

                // Use guest_memory.write to copy back safely
                guest_memory
                    .write_range_from_atomic(
                        &mem.subrange((dest_page * PAGE_SIZE64 + offset_in_page).try_into().unwrap(), dest_len),
                        src_slice,
                    )
                    .map_err(|_| DmaError::UnmapFailed)?;

                // Store the bounce buffer for later deallocation
                bounce_buffers.push(DmaBuffer {
                    id: transaction.id(),
                    offset: src_offset,
                    size: src_len,
                });
            } else if transaction.options().is_tx
                && transaction.backing() == MemoryBacking::BounceBuffer
            {
                bounce_buffers.push(DmaBuffer {
                    id: transaction.id(),
                    offset: transaction.offset(),
                    size: transaction.size() as usize,
                });
            }

            // Collect ranges for unmapping if pinned
            if transaction.backing() == MemoryBacking::Pinned {
                ranges.push(MemoryRange::new(
                    transaction.original_addr()
                        ..transaction.original_addr() + transaction.size() as u64,
                ));
            }
        }

        // Unmap the DMA transactions
        if !ranges.is_empty() {
            self._dma_manager_inner
                .lock()
                .manager_unmap_dma_transaction(&ranges)
                .map_err(|_| DmaError::UnmapFailed)?;
        }

        // Free all bounce buffers after copying data back
        for buffer in bounce_buffers {
            self.free(buffer);
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct DmaClientSpawner {
    dma_manager_inner: Arc<Mutex<GlobalDmaManagerInner>>,
}

impl DmaClientSpawner {
    /// Creates a new DMA client with the given device name.
    pub fn create_client(&self, device_name: String) -> anyhow::Result<Arc<DmaClientImpl>> {
        GlobalDmaManager::create_client_internal(&self.dma_manager_inner, device_name)
    }
}

struct BounceBufferAllocator {
    free_list: Vec<DmaBuffer>,
    allocated: Vec<DmaBuffer>,
    next_id: u64,
}
impl BounceBufferAllocator {
    fn new(size: usize) -> Self {
        let free_list = vec![DmaBuffer {
            id: 0,
            offset: 0,
            size,
        }];
        Self {
            free_list,
            allocated: Vec::new(),
            next_id: 1,
        }
    }

    /// Attempt to allocate a block of at least `size` bytes.
    /// Returns the offset into the MemoryBlock if successful, or None if not.
    fn malloc(&mut self, size: usize) -> Option<(u64, usize)> {
        if let Some(pos) = self.free_list.iter().position(|block| block.size >= size) {
            let block = self.free_list.remove(pos);
            let allocated_id = self.next_id;

            self.next_id = self.next_id.wrapping_add(1);

            let allocated_block = DmaBuffer {
                id: allocated_id,
                offset: block.offset,
                size,
            };

            self.allocated.push(allocated_block);
            self.allocated.sort_by_key(|b| b.offset); // Keep allocations sorted

            // If there’s remaining space, add back the unallocated portion
            if block.size > size {
                self.free_list.insert(
                    pos,
                    DmaBuffer {
                        id: 0,
                        offset: block.offset + size,
                        size: block.size - size,
                    },
                );
            }

            Some((allocated_id, block.offset))
        } else {
            None
        }
    }

    /// Free a previously allocated block.
    fn free(&mut self, id: u64) {
        if let Some(pos) = self.allocated.iter().position(|block| block.id == id) {
            let freed_block = self.allocated.remove(pos);
            self.insert_into_free_list(freed_block);
        }
    }

    fn insert_into_free_list(&mut self, new_block: DmaBuffer) {
        // Insert the new block in the sorted list using binary search
        let pos = match self
            .free_list
            .binary_search_by_key(&new_block.offset, |b| b.offset)
        {
            Ok(pos) | Err(pos) => pos, // `Err(pos)` gives us the insertion point
        };
        self.free_list.insert(pos, new_block);

        // Merge with the previous block if adjacent
        if pos > 0
            && self.free_list[pos - 1].offset + self.free_list[pos - 1].size
                == self.free_list[pos].offset
        {
            self.free_list[pos - 1].size += self.free_list[pos].size;
            self.free_list.remove(pos);
        }

        // Merge with the next block if adjacent
        if pos < self.free_list.len() - 1
            && self.free_list[pos].offset + self.free_list[pos].size
                == self.free_list[pos + 1].offset
        {
            self.free_list[pos].size += self.free_list[pos + 1].size;
            self.free_list.remove(pos + 1);
        }

        // Check if the last block can be merged with the preceding block if the new block was inserted at the end
        if pos == self.free_list.len() - 1 && self.free_list.len() > 1 {
            if self.free_list[pos - 1].offset + self.free_list[pos - 1].size
                == self.free_list[pos].offset
            {
                self.free_list[pos - 1].size += self.free_list[pos].size;
                self.free_list.remove(pos);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_initialization() {
        let allocator = BounceBufferAllocator::new(1024);
        assert_eq!(allocator.free_list.len(), 1);
        assert_eq!(allocator.free_list[0].size, 1024);
        assert_eq!(allocator.allocated.len(), 0);
    }

    #[test]
    fn test_malloc_success() {
        let mut allocator = BounceBufferAllocator::new(1024);
        let result = allocator.malloc(256);
        assert!(result.is_some());
        let (_id, offset) = result.unwrap();
        assert_eq!(offset, 0);
        assert_eq!(allocator.allocated.len(), 1);
        assert_eq!(allocator.free_list.len(), 1);
        assert_eq!(allocator.free_list[0].offset, 256);
        assert_eq!(allocator.free_list[0].size, 768);
    }

    #[test]
    fn test_malloc_fail() {
        let mut allocator = BounceBufferAllocator::new(512);
        let result = allocator.malloc(1024);
        assert!(result.is_none());
    }

    #[test]
    fn test_free_merging() {
        let mut allocator = BounceBufferAllocator::new(1024);
        let (id1, _) = allocator.malloc(256).unwrap();
        let (id2, _) = allocator.malloc(256).unwrap();
        allocator.free(id1);
        allocator.free(id2);
        assert_eq!(allocator.free_list.len(), 1);
        assert_eq!(allocator.free_list[0].size, 1024);
    }

    #[test]
    fn test_id_wrap_around() {
        let mut allocator = BounceBufferAllocator::new(1024);
        allocator.next_id = u64::MAX;
        let (id, _) = allocator.malloc(256).unwrap();
        assert_eq!(id, u64::MAX);
        assert_eq!(allocator.next_id, 0); // Wrapped around to 0
    }

    #[test]
    fn test_free_list_merging_with_adjacent_blocks() {
        let mut allocator = BounceBufferAllocator::new(1024);
        let (id1, _) = allocator.malloc(256).unwrap();
        let (id2, _) = allocator.malloc(256).unwrap();
        allocator.free(id1);
        allocator.free(id2);
        assert_eq!(allocator.free_list.len(), 1);
        assert_eq!(allocator.free_list[0].size, 1024);
    }

    #[test]
    fn test_out_of_order_allocation_and_free() {
        let mut allocator = BounceBufferAllocator::new(1024);

        let (_, offset1) = allocator.malloc(256).unwrap();
        let (id2, offset2) = allocator.malloc(512).unwrap();
        let (_, offset3) = allocator.malloc(128).unwrap();

        assert_eq!(offset1, 0);
        assert_eq!(offset2, 256);
        assert_eq!(offset3, 768);

        allocator.free(id2); // Free the middle block
        assert_eq!(allocator.free_list.len(), 2);

        let (_, offset4) = allocator.malloc(512).unwrap();
        assert_eq!(offset4, 256); // Should reuse freed block
    }

    #[test]
    fn test_freeing_non_contiguous_blocks() {
        let mut allocator = BounceBufferAllocator::new(1024);

        let (id1, _) = allocator.malloc(128).unwrap();
        let (_) = allocator.malloc(256).unwrap();
        let (id3, _) = allocator.malloc(128).unwrap();

        allocator.free(id1);
        allocator.free(id3);

        assert_eq!(allocator.free_list.len(), 2); // Two freed blocks should be separate
    }

    #[test]
    fn test_merge_blocks_after_out_of_order_free() {
        let mut allocator = BounceBufferAllocator::new(1024);

        let (id1, _) = allocator.malloc(256).unwrap();
        let (id2, _) = allocator.malloc(256).unwrap();
        let (id3, _) = allocator.malloc(512).unwrap();

        allocator.free(id1);
        allocator.free(id3);
        allocator.free(id2); // This should merge with previous free blocks

        assert_eq!(allocator.free_list.len(), 1); // Entire buffer should be free again
        assert_eq!(allocator.free_list[0].size, 1024);
    }
}
