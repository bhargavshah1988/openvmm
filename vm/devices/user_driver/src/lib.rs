// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Infrastructure for implementing PCI drivers in user mode.

// UNSAFETY: Manual memory management around buffers and mmap.
#![expect(unsafe_code)]

use inspect::Inspect;
use interrupt::DeviceInterrupt;
use memory::MemoryBlock;
use memory_range::MemoryRange;
use std::sync::Arc;

pub mod backoff;
pub mod emulated;
pub mod interrupt;
pub mod lockmem;
pub mod memory;
pub mod vfio;

pub type DmaAllocator<T> = <T as DeviceBacking>::DmaAllocator;

/// An interface to access device hardware.
pub trait DeviceBacking: 'static + Send + Inspect {
    /// An object for accessing device registers.
    type Registers: 'static + DeviceRegisterIo + Inspect;
    /// An object for allocating host memory to share with the device.
    type DmaAllocator: 'static + HostDmaAllocator;

    /// Returns a device ID for diagnostics.
    fn id(&self) -> &str;

    /// Maps a BAR.
    fn map_bar(&mut self, n: u8) -> anyhow::Result<Self::Registers>;

    /// DMA Client for the device.
    fn dma_client(&self) -> Arc<dyn DmaClient>;

    /// Returns the maximum number of interrupts that can be mapped.
    fn max_interrupt_count(&self) -> u32;

    /// Maps a MSI-X interrupt for use, returning an object that can be used to
    /// wait for the interrupt to be signaled by the device.
    ///
    /// `cpu` is the CPU that the device should target with this interrupt.
    ///
    /// This can be called multiple times for the same interrupt without disconnecting
    /// previous mappings. The last `cpu` value will be used as the target CPU.
    fn map_interrupt(&mut self, msix: u32, cpu: u32) -> anyhow::Result<DeviceInterrupt>;
}

/// Access to device registers.
pub trait DeviceRegisterIo: Send + Sync {
    /// Returns the length of the register space.
    fn len(&self) -> usize;
    /// Reads a `u32` register.
    fn read_u32(&self, offset: usize) -> u32;
    /// Reads a `u64` register.
    fn read_u64(&self, offset: usize) -> u64;
    /// Writes a `u32` register.
    fn write_u32(&self, offset: usize, data: u32);
    /// Writes a `u64` register.
    fn write_u64(&self, offset: usize, data: u64);
}

pub trait HostDmaAllocator: Send + Sync {
    /// Allocate a new block using default allocation strategy.
    fn allocate_dma_buffer(&self, len: usize) -> anyhow::Result<MemoryBlock>;
    /// Attach to a previously allocated memory block with contiguous PFNs.
    fn attach_dma_buffer(&self, len: usize, base_pfn: u64) -> anyhow::Result<MemoryBlock>;
}

#[derive(Debug)]
pub enum DmaError {
    InitializationFailed,
    MapFailed,
    UnmapFailed,
    PinFailed,
    BounceBufferFailed,
}
/// Enum representing memory backing type.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum MemoryBacking {
    Pinned,
    PrePinned,
    BounceBuffer,
    Physical,
}
#[derive(Debug, Clone)]
pub struct DmaTransectionOptions {
    pub force_bounce_buffer: bool, // Always use bounce buffers, even if pinning succeeds
}
pub struct DmaTransactionHandler {
    pub transactions: Vec<DmaTransaction>,
}

pub struct ContiguousBuffer {
    offset: u32,
    len: u64,
    padding_len: u32,
    committed: bool,
    pub gpa: u64,
}

pub struct DmaTransaction {
    dma_buffer: ContiguousBuffer,
    original_addr: u64,
    backing: MemoryBacking,
}
impl DmaTransaction {
    /// Creates a new `DmaTransaction` with controlled field access
    pub fn new(dma_buffer: ContiguousBuffer, original_addr: u64, backing: MemoryBacking) -> Self {
        Self {
            dma_buffer,
            original_addr,
            backing,
        }
    }
    /// Public getter methods
    pub fn dma_addr(&self) -> u64 {
        self.dma_buffer.gpa
    }
    pub fn size(&self) -> u64 {
        self.dma_buffer.len
    }
    pub fn backing(&self) -> MemoryBacking {
        self.backing
    }
    pub fn original_addr(&self) -> u64 {
        self.original_addr
    }
}

pub trait DmaClient: Send + Sync {
    /// Allocate a new DMA buffer.
    fn allocate_dma_buffer(&self, total_size: usize) -> anyhow::Result<MemoryBlock>;

    /// Attach to a previously allocated memory block
    fn attach_dma_buffer(&self, len: usize, base_pfn: u64) -> anyhow::Result<MemoryBlock>;

    fn map_dma_ranges(
        &self,
        ranges: &[MemoryRange],
        options: Option<&DmaTransectionOptions>,
    ) -> Result<DmaTransactionHandler, DmaError>;


    fn unmap_dma_ranges(&self, dma_transactions: &[DmaTransaction]) -> Result<(), DmaError>;
}

impl ContiguousBuffer {
    pub fn new(offset: u32, len: u64, padding_len: u32, gpa: u64) -> Self {
        Self {
            offset,
            len,
            padding_len,
            committed: false,
            gpa,
        }
    }
}
