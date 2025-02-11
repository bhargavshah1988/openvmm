// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! This module provides a global DMA manager and client implementation.
//! It manages DMA buffers and provides clients with access to these buffers.
//! The `GlobalDmaManager` creates DMA buffers for different devices.
//! The `DmaClientImpl` struct implements the `user_driver::DmaClient` trait.

use event_listener::Event;
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

    //fn write(&mut self, buffer: DmaBuffer, data: &[u8]) -> anyhow::Result<()> {
    //    if buffer.size < data.len() {
    //        return Err(anyhow::anyhow!("Buffer size is smaller than data length"));
    //    }

    //    let start_offset = buffer.offset;
    //    let end_offset = start_offset + data.len();

    //    if end_offset > self.mem.len() {
    //        return Err(anyhow::anyhow!("Write exceeds memory block bounds"));
    //    }

    //    let start_page = start_offset / PAGE_SIZE32 as usize;
    //    let offset_in_page = start_offset % PAGE_SIZE32 as usize;

    //    // Get the physical frame number (PFN)
    //    if let Some(&pfn) = self.mem.pfns().get(start_page as usize) {
    //        let gpa = pfn * PAGE_SIZE64 + offset_in_page as u64;
    //        self.mem.write_at(gpa as usize, data);
    //        Ok(())
    //    } else {
    //        Err(anyhow::anyhow!("Invalid memory page"))
    //    }
    //}

    //fn read(&self, buffer: &DmaBuffer, output: &mut [u8]) -> anyhow::Result<()> {
    //    if buffer.size < output.len() {
    //        return Err(anyhow::anyhow!("Buffer size is smaller than requested read length"));
    //    }

    //    let start_page = buffer.offset / PAGE_SIZE32 as usize;
    //    let offset_in_page = buffer.offset % PAGE_SIZE32 as usize;

    //    self.mem.read_at(start_page * PAGE_SIZE32 as usize + offset_in_page, output);
    //    Ok(())
    //}
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
        let ranges = &mem.memoryranges();
        let dma_transaction_handler;
        let mut transactions = Vec::new();

        if options.is_rx {
            let bounce_buffer = self.allocate(mem.len());
            let mut cumulative_offset: usize = 0;

            let dma_transactions: Vec<DmaTransaction> = ranges
                .iter()
                .map(|range| {
                    let range_size = range.len(); // u64
                    let transaction_offset = bounce_buffer.offset + cumulative_offset;

                    let contig_buf = DmaBuffer::new(0, transaction_offset, range_size as usize);

                    cumulative_offset += range_size as usize;

                    // Remove the semicolon so the closure returns a `DmaTransaction`
                    DmaTransaction::new(
                        contig_buf,
                        range.start(),
                        options.clone(),
                        MemoryBacking::BounceBuffer,
                    )
                })
                .collect();

            dma_transaction_handler = user_driver::DmaTransactionHandler {
                transactions: dma_transactions,
            };

            // Ensure early return to match function signature
            return Ok(dma_transaction_handler);
        }

        match self
            ._dma_manager_inner
            .lock()
            .manager_map_dma_transaction(ranges)
        {
            Ok(_) => {
                let dma_transactions = ranges
                    .iter()
                    .map(|range| {
                        let contig_buf = DmaBuffer::new(0, 0, range.len().try_into().unwrap());

                        DmaTransaction::new(
                            contig_buf,
                            range.start(),
                            options.clone(),
                            MemoryBacking::Pinned,
                        )
                    })
                    .collect();

                dma_transaction_handler = user_driver::DmaTransactionHandler {
                    transactions: dma_transactions,
                };
            }
            Err(_) => {
                // Mapping failed, allocate a bounce buffer
                let mut total_size = mem.len();
                let bounce_buffer = self.allocate(total_size);

                let mut offset = 0;
                let page_count = bounce_buffer.size / PAGE_SIZE64 as usize;

                let mut remaining = total_size;
                for i in 0..page_count {
                    // Determine how many bytes to copy for this page.
                    let len = PAGE_SIZE32.min(remaining.try_into().unwrap()) as usize;
                    remaining -= len as usize;

                    // Compute the effective destination offset within the overall bounce buffer.
                    // For page i, the effective offset is:
                    let effective_offset = bounce_buffer.offset + i * PAGE_SIZE32 as usize;

                    // Convert the effective offset into a destination page and an offset within that page.
                    let dest_page = effective_offset / PAGE_SIZE32 as usize;
                    let offset_in_page = effective_offset - dest_page * PAGE_SIZE32 as usize;

                    // Compute the destination address using the parent's MemoryBlock PFNs.
                    // Here we use the PFN for dest_page, then add the offset within that page.
                    let dest_addr =
                        self.mem.pfns()[dest_page] * PAGE_SIZE64 + offset_in_page as u64;

                    // Copy from the guest memory subrange into the destination slice.
                    // We assume that `mem.subrange(offset, len)` returns the guest subrange starting at the given offset with the specified length.
                    // And self.mem.as_mut_slice() gives us mutable access to the underlying MemoryBlock.
                    guest_memory
                        .read_range_to_atomic(
                            &mem.subrange(i * PAGE_SIZE32 as usize, len),
                            &self.mem.as_slice()[dest_page * PAGE_SIZE as usize..][..len],
                        )
                        .map_err(|e| DmaError::BounceBufferFailed)?;
                }

                // Calculate the total size of the combined ranges
                let total_size: u64 = ranges.iter().map(|range| range.len() as u64).sum();

                // Create a single DmaBuffer that spans all the ranges
                let combined_buffer = DmaBuffer::new(bounce_buffer.id, bounce_buffer.offset, total_size as usize);

                // Set up for DMA transactions
                let mut cumulative_offset: usize = 0; // To track the offset within the combined buffer
                for range in ranges {
                    // Calculate the size for the current range
                    let range_size = range.len() as u64;

                    // Calculate the offset for the current range within the combined buffer
                    let transaction_offset = combined_buffer.offset + cumulative_offset;

                    // Create a DMA transaction for the current range using the same combined buffer
                    transactions.push(DmaTransaction::new(
                        combined_buffer.clone(), // Use the same buffer for all ranges
                        range.start(),
                        options.clone(),
                        MemoryBacking::BounceBuffer,
                    ));

                    // Update the cumulative offset for the next range
                    cumulative_offset += range_size as usize;
                }
                dma_transaction_handler = user_driver::DmaTransactionHandler { transactions };
            }
        }

        Ok(dma_transaction_handler)
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
            if transaction.options().is_rx  && transaction.backing() == MemoryBacking::BounceBuffer {
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
                let dest_page = src_offset / PAGE_SIZE32 as usize;
                let offset_in_page = src_offset % PAGE_SIZE32 as usize;

                // Use guest_memory.write to copy back safely
                guest_memory
                    .write_range_from_atomic(
                        &mem.subrange(dest_page * PAGE_SIZE32 as usize + offset_in_page, dest_len),
                        src_slice,
                    )
                    .map_err(|_| DmaError::UnmapFailed)?;

                // Store the bounce buffer for later deallocation
                bounce_buffers.push(DmaBuffer{
                    id: transaction.id(),
                    offset: src_offset,
                    size: src_len,
                });
            }
            else if transaction.options().is_tx && transaction.backing() == MemoryBacking::BounceBuffer {
                bounce_buffers.push(DmaBuffer{
                    id: transaction.id(),
                    offset:transaction.offset(),
                    size:transaction.size() as usize,
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
        let free_list = vec![DmaBuffer { id: 0, offset: 0, size }];
        Self { free_list, allocated: Vec::new(), next_id: 1 }
    }

    /// Attempt to allocate a block of at least `size` bytes.
    /// Returns the offset into the MemoryBlock if successful, or None if not.
    fn malloc(&mut self, size: usize) -> Option<(u64, usize)> {
        if let Some(pos) = self
            .free_list
            .iter()
            .position(|block| block.size >= size)
        {
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
                self.free_list.insert(pos, DmaBuffer {
                    id: 0,
                    offset: block.offset + size,
                    size: block.size - size
                });
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
        let pos = match self.free_list.binary_search_by_key(&new_block.offset, |b| b.offset) {
            Ok(pos) | Err(pos) => pos, // `Err(pos)` gives us the insertion point
        };
        self.free_list.insert(pos, new_block);

        // Merge with the previous block if adjacent
        if pos > 0 && self.free_list[pos - 1].offset + self.free_list[pos - 1].size == self.free_list[pos].offset {
            self.free_list[pos - 1].size += self.free_list[pos].size;
            self.free_list.remove(pos);
        }

        // Merge with the next block if adjacent
        if pos < self.free_list.len() - 1
            && self.free_list[pos].offset + self.free_list[pos].size == self.free_list[pos + 1].offset
        {
            self.free_list[pos].size += self.free_list[pos + 1].size;
            self.free_list.remove(pos + 1);
        }

        // Check if the last block can be merged with the preceding block if the new block was inserted at the end
        if pos == self.free_list.len() - 1 && self.free_list.len() > 1 {
            if self.free_list[pos - 1].offset + self.free_list[pos - 1].size == self.free_list[pos].offset {
                self.free_list[pos - 1].size += self.free_list[pos].size;
                self.free_list.remove(pos);
            }
        }
    }

}
