// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! This module provides a global DMA manager and client implementation.
//! It manages DMA buffers and provides clients with access to these buffers.
//! The `GlobalDmaManager` creates DMA buffers for different devices.
//! The `DmaClientImpl` struct implements the `user_driver::DmaClient` trait.

use parking_lot::Mutex;
use user_driver::DmaTransaction;
use std::sync::Arc;
use user_driver::memory::MemoryBlock;
use user_driver::vfio::VfioDmaBuffer;
use user_driver::MemoryBacking;
use virt_mshv_vtl::UhPartition;
use memory_range::MemoryRange;

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
                .map_err(|e| anyhow::anyhow!("Failed to get DMA buffer allocator: {:?}", e))?
        };

        let client = DmaClientImpl {
            _dma_manager_inner: inner.clone(),
            dma_buffer_allocator: Some(allocator.clone()), // Set the allocator now
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
                    range.start(),
                    range.len(),
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
