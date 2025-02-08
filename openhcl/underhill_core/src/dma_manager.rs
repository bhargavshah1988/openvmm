// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! This module provides a global DMA manager and client implementation.
//! It manages DMA buffers and provides clients with access to these buffers.
//! The `GlobalDmaManager` creates DMA buffers for different devices.
//! The `DmaClientImpl` struct implements the `user_driver::DmaClient` trait.

use parking_lot::Mutex;
use std::sync::Arc;
use user_driver::memory::MemoryBlock;
use user_driver::vfio::VfioDmaBuffer;
use user_driver::memory::PAGE_SIZE64;
use user_driver::memory::PAGE_SIZE32;
use std::sync::atomic::AtomicU8;
use event_listener::Event;
use virt_mshv_vtl::UhPartition;
use memory_range::MemoryRange;
use user_driver::DmaTransaction;
use user_driver::ContiguousBuffer;
use user_driver::MemoryBacking;

pub struct GlobalDmaManager {
    inner: Arc<Mutex<GlobalDmaManagerInner>>,
}

pub struct GlobalDmaManagerInner {
    dma_buffer_spawner: Box<dyn Fn(String) -> anyhow::Result<Arc<dyn VfioDmaBuffer>> + Send>,
    partition: Arc<UhPartition>,
}

impl GlobalDmaManagerInner {
    pub fn manager_map_dma_transaction(&self, ranges: &[MemoryRange]) -> anyhow::Result<()> {
        self.
        partition
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
        let inner = GlobalDmaManagerInner { dma_buffer_spawner, partition };

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

        let mem = allocator.create_dma_buffer((100 * PAGE_SIZE32).try_into().unwrap())
            .map_err(|e| anyhow::anyhow!("Failed to create DMA buffer: {:?}", e))?;

        let client = DmaClientImpl {
            _dma_manager_inner: inner.clone(),
            dma_buffer_allocator: Some(allocator.clone()),
            buffer_manager : ContiguousBufferManager::new(mem)?,
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
    buffer_manager: ContiguousBufferManager,
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
        ranges: &[MemoryRange],
        options: Option<&user_driver::DmaTransectionOptions>,
    ) -> Result<user_driver::DmaTransactionHandler, user_driver::DmaError> {

        self._dma_manager_inner
            .lock()
            .manager_map_dma_transaction(ranges)
            .map_err(|_| user_driver::DmaError::MapFailed)?;

        let dma_transactions = ranges
            .iter()
            .map(|range| {
                DmaTransaction::new(
                    ContiguousBuffer::new(
                        0,
                        range.len(),
                        0,
                        range.start(),
                    ),
                    range.start(),
                    MemoryBacking::Pinned,
                )
            })
            .collect();
        Ok(user_driver::DmaTransactionHandler {
            transactions: dma_transactions,
        })
    }

    fn unmap_dma_ranges(
        &self,
        dma_transactions: &[DmaTransaction],
    ) -> Result<(), user_driver::DmaError> {
    let ranges: Vec<MemoryRange> = dma_transactions
        .iter()
        .filter_map(|transaction| {
            if transaction.backing() == MemoryBacking::Pinned {
                Some(MemoryRange::new(transaction.original_addr()..transaction.original_addr() + transaction.size()))
            } else {
                None
            }
        })
        .collect();
    if !ranges.is_empty() {
        self._dma_manager_inner
            .lock()
            .manager_unmap_dma_transaction(&ranges)
            .map_err(|_| user_driver::DmaError::UnmapFailed)?;
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

pub(crate) struct ContiguousBufferManager {
    core: Mutex<ContiguousBufferCore>,
    mem: MemoryBlock,
    event: Event,
}

#[derive(Debug, thiserror::Error)]
#[error("out of contiguous buffer memory")]
struct OutOfMemory;

impl ContiguousBufferManager {
    pub fn new(mem: MemoryBlock) -> Result<Self, anyhow::Error> {
        let len = mem.len() as u32;
        Ok(Self {
            core: Mutex::new(ContiguousBufferCore::new(len)),
            mem,
            event: Event::new(),
        })
    }

    pub async fn allocate(&self, len: u32) -> Result<ContiguousBuffer, OutOfMemory> {
        loop {
            let mut core = self.core.lock();
            if let Some(buffer) = core.try_allocate(&self.mem, len) {
                return Ok(buffer);
            }
            drop(core);
            self.event.listen().await;
        }
    }

    pub fn free(&self, offset: u32, len_with_padding: u32) {
        let mut core = self.core.lock();
        core.free(offset, len_with_padding);
        drop(core);
        self.event.notify(1);
    }

    pub fn as_slice(&self) -> &[AtomicU8] {
        self.mem.as_slice()
    }
}

struct ContiguousBufferCore {
    len: u32,
    head: u32,
    tail: u32,
}

impl ContiguousBufferCore {
    fn new(len: u32) -> Self {
        Self {
            len,
            head: 0,
            tail: len - 1,
        }
    }

    fn try_allocate(&mut self, mem: &MemoryBlock, len: u32) -> Option<ContiguousBuffer> {
        let mut allocated_offset = self.head;
        let mut len_with_padding = len;
        let bytes_remaining_on_page = PAGE_SIZE32 - (allocated_offset % PAGE_SIZE32);

        // Skip to next page if needed
        let min_usable_bytes = 1500;
        if len > bytes_remaining_on_page && bytes_remaining_on_page < min_usable_bytes {
            allocated_offset = allocated_offset.wrapping_add(bytes_remaining_on_page);
            len_with_padding += bytes_remaining_on_page;
        }

        let available_space = if self.head < self.tail {
            self.tail - self.head
        } else {
            self.len - (self.head - self.tail)
        };

        if len_with_padding > available_space {
            return None;
        }

        self.head = self.head.wrapping_add(len_with_padding);

        let start_page = allocated_offset / PAGE_SIZE32;
        let end_page = (allocated_offset + len_with_padding) / PAGE_SIZE32;
        let offset_in_page = allocated_offset % PAGE_SIZE32;

        // Compute GPA
        let mut gpa = 0;
        for page in start_page..end_page {
            if let Some(&pfn) = mem.pfns().get(page as usize) {
                if page == start_page {
                    gpa = pfn * PAGE_SIZE64 + offset_in_page as u64;
                }
            } else {
                return None;
            }
        }

        Some(ContiguousBuffer::new(
            allocated_offset % self.len,
            len as u64,
            len_with_padding - len,
            gpa,
        ))
    }

    fn free(&mut self, _offset: u32, len_with_padding: u32) {
        self.tail = self.tail.wrapping_add(len_with_padding);
    }
}


