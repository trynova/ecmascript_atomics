// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Copyright of the originating code is owned by Firefox authors and Mozilla,
// modifications by Aapo Alasuutari.

#![no_std]

//! # Racy atomic operations
//!
//! This library provides atomic operations that match the ECMAScript
//! specification's memory model: this is effectively the same memory model as
//! Java's shared variables model. This model allows non-atomic reads and
//! writes to perform data races, and also allows mixed-size atomic reads and
//! writes. Both of these are undefined behaviour in the C++/Rust memory model,
//! which is why these atomic operations are implemented using inline assembly.
//!
//! The contents of the library are copied and adapted from Mozilla Firefox's
//! source code, basing mainly on [GenerateAtomicOperations.py].
//!
//! The fundamental constraints on the functions are:
//!
//! - That any Atomic operations performed on memory used in these
//!   functions must only be accessed using these functions or functions
//!   that have a compatible realisation. Importantly, it is strictly
//!   undefined behaviour to use normal Rust Atomics in conjunction with
//!   the racy atomic operations in this library.
//!
//! - That accesses may race without creating C++/Rust undefined behavior:
//!   atomic accesses (marked "SeqCst") may race with non-atomic
//!   accesses (marked "Unordered"); overlapping but non-matching,
//!   and hence incompatible, atomic accesses may race; and non-atomic
//!   accesses may race. The effects of races need not be predictable,
//!   so garbage can be produced by a read or written by a write, but
//!   the effects must be benign: the program must continue to run, and
//!   only the memory in the union of addresses named in the racing
//!   accesses may be affected.
//!
//! The compatibility constraint means that if the memory is accessed elsewhere
//! then it must make compatible decisions about how to implement atomic
//! operations with the functions below.
//!
//! The safe-for-races constraint means that by and large, it is hard or
//! impossible to implement these primitives in C++/Rust. See "Implementation
//! notes" below.
//!
//! The "SeqCst" suffix on operations means "sequentially consistent"
//! and means such a function's operation must have "sequentially
//! consistent" memory ordering. This corresponds with the Rust Atomics'
//! "SeqCst" ordering.
//!
//! Note that an "Unordered" access does not provide the atomicity of
//! a "relaxed atomic" access: it can read or write garbage if there's
//! a race.
//!
//!
//! ## Implementation notes.
//!
//! It's not a requirement that these functions be inlined.
//!
//! In principle these functions will not be written in Rust, thus
//! making races defined behavior if all racy accesses from Rust go via
//! these functions.
//!
//! The appropriate implementations will be platform-specific and
//! there are some obvious implementation strategies to choose
//! from, sometimes a combination is appropriate:
//!
//!  - generating the code at run-time with a JIT;
//!  - hand-written assembler (maybe inline); or
//!  - using special compiler intrinsics or directives.
//!
//! Trusting the compiler not to generate code that blows up on a
//! race definitely won't work in the presence of TSan, or even of
//! optimizing compilers in seemingly-"innocuous" conditions. (See
//! https://www.usenix.org/legacy/event/hotpar11/tech/final_files/Boehm.pdf
//! for details.)
//!
//! [GenerateAtomicOperations.py]: https://searchfox.org/firefox-main/source/js/src/jit/GenerateAtomicOperations.py

mod generated;
mod unordered_copy;

use core::{
    cell::UnsafeCell, hint::assert_unchecked, marker::PhantomData, mem::MaybeUninit, ptr::NonNull,
};

use generated::*;
use unordered_copy::*;

use crate::private::Sealed;

/// Public trait for allowing users of the library to name the types that can
/// be natively stored in racy memory. These are effectively the unsigned
/// integers from 1 to 8 bytes in size.
pub trait RacyStorage: Sealed {}

impl RacyStorage for u8 {}
impl RacyStorage for u16 {}
impl RacyStorage for u32 {}
impl RacyStorage for u64 {}
impl RacyStorage for usize {}

/// ECMAScript atomic memory orderings
///
/// Memory orderings specify the way atomic operations synchronise memory.
/// With [`Ordering::Unordered`], no synchronisation is performed. With
/// [`Ordering::SeqCst`], a store-load pair of operations synchronize other
/// memory while additionally preserving a total order of such operations
/// across all threads.
///
/// The ECMAScript memory model is explained in the [ECMAScript Language
/// specification](https://tc39.es/ecma262/#sec-memory-model). Note that the
/// "INIT" ordering is not offered here as all initialisation of the memory
/// should happen before calling the [`enter`] method used to move memory from
/// the Rust memory model over into the ECMAScript memory model.
///
/// [`enter`]: crate::RacyMemory::enter
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Ordering {
    /// No ordering constraints, stores and loads may tear (ie. are not
    /// necessarily atomic).
    ///
    /// Corresponds to [`Unordered`] in LLVM.
    ///
    /// [`Unordered`]: https://llvm.org/docs/Atomics.html#unordered
    Unordered,
    /// All threads see all seqeuentially consistent operations in the same
    /// order.
    ///
    /// Corresponds to [`SeqCst`] in Rust.
    ///
    /// [`SeqCst`]: https://doc.rust-lang.org/std/sync/atomic/enum.Ordering.html#variant.SeqCst
    SeqCst,
}

/// A sequentially consistent atomic fence.
///
/// See [std::sync::atomic::fence] for details.
#[inline(always)]
pub fn fence() {
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

/// Emits a machine instruction to signal the processor that it is running in a
/// busy-wait spin-loop (“spin lock”).
///
/// See [std::hint::spin_loop] for details.
#[inline(always)]
pub fn atomic_pause() {
    core::hint::spin_loop();
}

/// Opaque handle to a slab of memory with the ECMAScript Atomics memory model.
/// The slab must be created using the [`enter`] method and must be turned back
/// into Rust memory using the [`exit`] method (note; this must be strictly
/// synchronised with all possible users of the racy atomic memory).
///
/// The memory can be freely shared to multiple threads for use and all APIs on
/// the memory are guaranteed to not cause undefined behaviour even when data
/// races or mixed-size atomics are used. Tearing may occur when using the copy
/// APIs, meaning that usage should generally be careful and try to avoid both
/// data races and mixed-size atomics.
///
/// [`enter`]: crate::RacyMemory::enter
/// [`exit`]: crate::RacyMemory::exit
///
/// # Soundness
///
/// The memory behind this handle is not and must not be read as Rust memory.
/// Any Rust reads or writes into the memory are undefined behaviour.
///
/// # Allocations
///
/// The [`enter`] method takes ownership of Rust memory and (conceptually)
/// deallocates it. Therefore, the pointer passed into the method must not be
/// deallocated by the caller. Conversely, the [`exit`] method (conceptually)
/// allocates new Rust memory and returns a pointer to it to the caller. The
/// caller is therefore responsible for deallocating the Rust memory.
///
/// Each `enter` call must be matched by an equal `exit` call on the same
/// slice, lest the ECMAScript memory be leaked.
///
/// [`enter`]: crate::RacyMemory::enter
/// [`exit`]: crate::RacyMemory::exit
pub struct RacyMemory<T: RacyStorage> {
    ptr: RacyPtr<T>,
    len: usize,
}

// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Send for RacyMemory<T> {}
// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Sync for RacyMemory<T> {}

impl<T: RacyStorage> RacyMemory<T> {
    /// Move a memory allocation into the ECMAScript memory model.
    ///
    /// # Safety
    ///
    /// `ptr` must be a pointer to `len` bytes of readable and writable memory.
    ///
    /// # Soundness
    ///
    /// This deallocates the existing memory behind `ptr` (in an abstract
    /// machine sense), such that after this call the `ptr` must be viewed as
    /// an invalid pointer. It is thus strictly forbidden for any Rust code to
    /// read from or write into the memory, including with atomic operations.
    #[inline]
    #[must_use]
    pub unsafe fn enter(ptr: NonNull<u8>, len: usize) -> RacyMemory<u8> {
        let mut ptr = ptr.as_ptr();
        // SAFETY: noop.
        unsafe {
            core::arch::asm!(
                "/* Magic spell: let {} no longer be memory in Rust's eyes! */",
                // Note: ptr is and out parameter so that the assembly block
                // can conceptually deallocate the original ptr, removing its
                // provenance, and return a new ptr with difference provenance.
                inlateout(reg) ptr,
                options(nostack, preserves_flags)
            )
        }
        // SAFETY: Magic spell always returns non-null pointers.
        let ptr = RacyPtr::from_ptr(unsafe { NonNull::new_unchecked(ptr.cast()) });
        RacyMemory::from_raw_parts(ptr, len)
    }

    /// Move a typed memory allocation into the ECMAScript memory model.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid, uniquely owned `T`.
    ///
    /// # Soudness
    ///
    /// See [`enter`] for details.
    ///
    /// [`enter`]: crate::RacyMemory::enter
    pub unsafe fn enter_ptr(ptr: NonNull<T>) -> Self {
        let RacyMemory { ptr, .. } = unsafe { Self::enter(ptr.cast(), size_of::<T>()) };
        Self::from_raw_parts(ptr.cast(), 1)
    }

    /// Move a typed memory allocation slice into the ECMAScript memory model.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid, uniquely owned `T`.
    ///
    /// # Soudness
    ///
    /// See [`enter`] for details.
    ///
    /// [`enter`]: crate::RacyMemory::enter
    pub unsafe fn enter_slice(ptr: NonNull<[T]>) -> Self {
        let len = ptr.len();
        let RacyMemory { ptr, .. } = unsafe { Self::enter(ptr.cast(), size_of::<T>() * len) };
        Self::from_raw_parts(ptr.cast(), len)
    }

    /// Move an ECMAScript memory model into the void and return a new Rust
    /// memory pointer to `len` bytes.
    ///
    /// # Safety
    ///
    /// This must be the only referrer to the racy atomic memory.
    ///
    /// # Soundness
    ///
    /// This deallocates the ECMAScript memory behind `ptr` (in an abstract
    /// sense). It is thus strictly forbidden to use any racy value derived
    /// from this memory after this call happens, on any thread. This call must
    /// therefore be strictly synchronised between threads, and only one thread
    /// is allowed to perform this call.
    #[inline]
    #[must_use]
    pub unsafe fn exit(self) -> (NonNull<T>, usize) {
        let mut ptr = self.ptr.as_ptr().as_ptr();
        // SAFETY: noop.
        unsafe {
            core::arch::asm!(
                "/* Magic spell: let {} be memory in Rust's eyes! */",
                // Note: ptr is and out parameter so that the assembly block
                // can conceptually deallocate the ECMAScript memory, allocate
                // new Rust memory, and return a pointer to it.
                inlateout(reg) ptr,
                options(nostack, preserves_flags)
            )
        }
        // SAFETY: Magic spell always returns non-null pointers.
        (unsafe { NonNull::new_unchecked(ptr.cast()) }, self.len)
    }

    /// Access the racy atomic memory using a shared slice.
    #[inline(always)]
    pub const fn as_slice(&self) -> RacySlice<'_, T> {
        // SAFETY: type guarantees proper allocation.
        unsafe { RacySlice::from_raw_parts(self.ptr, self.len) }
    }

    /// Destructure a racy atomic memory into raw parts.
    #[inline(always)]
    pub const fn into_raw_parts(self) -> (RacyPtr<T>, usize) {
        (self.ptr, self.len)
    }

    /// Recrate a racy atomic memory slice from raw parts.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a racy atomic memory with a length of exactly `len`
    /// elements and be properly aligned. Note that this function does
    /// not "reallocate" the pointed-to memory.
    pub const fn from_raw_parts(ptr: RacyPtr<T>, len: usize) -> Self {
        Self { ptr, len }
    }
}

mod private {
    pub trait Sealed: 'static + Copy + Eq + Send + Sync + core::fmt::Display {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for usize {}
}

/// Opaque handle to a slice of memory with the ECMAScript Atomics memory
/// model.
///
/// # Soundness
///
/// The memory behind this handle is not and must not be read as Rust memory.
/// Any Rust reads or writes into the memory are undefined behaviour.
#[derive(Clone, Copy)]
pub struct RacySlice<'a, T: RacyStorage> {
    ptr: RacyPtr<T>,
    len: usize,
    __marker: PhantomData<&'a UnsafeCell<T>>,
}

// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Send for RacySlice<'_, T> {}
// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Sync for RacySlice<'_, T> {}

impl<'a, T: RacyStorage> RacySlice<'a, T> {
    /// Destructure a racy atomic memory slice into raw parts.
    #[inline(always)]
    pub const fn into_raw_parts(self) -> (RacyPtr<T>, usize) {
        (self.ptr, self.len)
    }

    /// Create a new racy atomic memory slice from a racy atomic byte pointer
    /// and length.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a racy atomic memory with a length of at least
    /// `len` elements and be properly aligned. Note that this function does
    /// not "reallocate" the pointed-to memory.
    #[inline(always)]
    pub const unsafe fn from_raw_parts(ptr: RacyPtr<T>, len: usize) -> Self {
        Self {
            ptr,
            len,
            __marker: PhantomData,
        }
    }

    /// Get the slice's internal pointer.
    const fn as_ptr(&self) -> RacyPtr<T> {
        self.ptr
    }

    /// Returns the number of elements in the slice.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return the byte length of the slice.
    const fn byte_length(&self) -> usize {
        // SAFETY: slice is guaranteed to point to len valid items.
        unsafe {
            assert_unchecked(self.len().checked_mul(size_of::<T>()).is_some());
            self.len.unchecked_mul(size_of::<T>())
        }
    }

    /// Returns `true` if the slice has a length of 0.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Transmutes the racy memory slice to a slice of another racy memory
    /// type, ensuring alignment of the types is maintained.
    ///
    /// This method splits the slice into three distinct slices: prefix,
    /// correctly aligned middle slice of a new type, and the suffix slice. The
    /// middle part will be as big as possible under the given alignment
    /// constraint and element size.
    pub fn align_to<U: RacyStorage>(
        self,
    ) -> (RacySlice<'a, u8>, RacySlice<'a, U>, RacySlice<'a, u8>) {
        let byte_length = self.byte_length();
        let head_ptr = self.ptr.as_ptr();
        let head_length = head_ptr.align_offset(align_of::<U>()).min(byte_length);
        // SAFETY: does not overflow length.
        let body_ptr = unsafe { head_ptr.byte_add(head_length) };
        let remaining_byte_length = byte_length - head_length;
        let body_length = remaining_byte_length / size_of::<U>();
        let tail_length = remaining_byte_length % size_of::<U>();
        debug_assert_eq!(
            head_length + body_length * size_of::<U>() + tail_length,
            byte_length,
            "align_to produced a collection of slices with different total byte length than original"
        );
        // SAFETY: cannot overflow len as body_length is derived from len.
        let tail_ptr = unsafe { body_ptr.byte_add(body_length * size_of::<U>()) };
        if const { align_of::<U>() <= align_of::<T>() } {
            // SAFETY: optimisation check.
            unsafe {
                assert_unchecked(
                    head_length == 0
                        && tail_length == 0
                        && body_length == byte_length / size_of::<U>(),
                );
            };
        }
        (
            RacySlice {
                ptr: RacyPtr::from_ptr(head_ptr),
                len: head_length,
                __marker: PhantomData,
            },
            RacySlice {
                ptr: RacyPtr::from_ptr(body_ptr),
                len: body_length,
                __marker: PhantomData,
            },
            RacySlice {
                ptr: RacyPtr::from_ptr(tail_ptr),
                len: tail_length,
                __marker: PhantomData,
            },
        )
    }

    /// Transmutes the the racy memory slice to a slice of racy bytes.
    #[inline(always)]
    pub const fn to_bytes(self) -> RacySlice<'a, u8> {
        // SAFETY: Totally safe here.
        unsafe { RacySlice::from_raw_parts(self.as_ptr().cast(), self.byte_length()) }
    }

    /// Creates an iterator from a value.
    #[inline]
    pub const fn iter(&self) -> RacyIter<'a, T> {
        RacyIter::new(*self)
    }

    /// Create a slice of racy atomic memory starting at the given offset.
    /// Returns an empty slice if the offset is beyond the end of this slice.
    #[inline]
    pub fn slice_from(&self, offset: usize) -> Self {
        // SAFETY: cannot overflow len.
        let ptr = RacyPtr::from_ptr(unsafe {
            self.ptr
                .as_ptr()
                .byte_add(offset.min(self.len) * size_of::<T>())
        });
        let len = self.len.saturating_sub(offset);
        // SAFETY: Guaranteed to be proper.
        unsafe { Self::from_raw_parts(ptr, len) }
    }

    /// Create a slice of racy atomic memory ending at the given offset.
    /// Returns self if the offset is beyond the end of this slice.
    #[inline]
    pub fn slice_to(&self, offset: usize) -> Self {
        let ptr = self.ptr;
        let len = self.len.min(offset);

        // SAFETY: Guaranteed to be proper.
        unsafe { Self::from_raw_parts(ptr, len) }
    }

    /// Create a slice of racy atomic memory with the given offsets, bounded to
    /// the end of this slice.
    #[inline]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        // First bound the offsets within the slice.
        let start = start.min(self.len);
        let end = end.min(self.len);
        // Then calculate the slice length, saturating at 0 if order is reversed.
        let len = end.saturating_sub(start);
        // SAFETY: cannot overflow len.
        let ptr = RacyPtr::from_ptr(unsafe { self.ptr.as_ptr().byte_add(start * size_of::<T>()) });
        // SAFETY: Guaranteed to be proper.
        unsafe { Self::from_raw_parts(ptr, len) }
    }

    /// Convert the memory into a racy atomic of given type at the start of the
    /// slice. Returns None if the slice is smaller than T bytes in size or is
    /// not correctly aligned.
    #[inline]
    pub fn as_aligned<U: RacyStorage>(&self) -> Option<Racy<'a, U>> {
        // SAFETY: Slice should always be aligned to its own type.
        unsafe { assert_unchecked(self.ptr.as_ptr().cast::<T>().is_aligned()) };
        if self.byte_length() < size_of::<U>() || !self.ptr.as_ptr().cast::<U>().is_aligned() {
            None
        } else {
            Some(Racy::from_ptr(self.ptr.as_ptr()))
        }
    }

    /// Load a value from the start of this slice using Unordered atomic
    /// ordering. The slice need not be aligned to the type.
    ///
    /// Returns None if the slice is smaller than the type.
    ///
    /// # Tearing
    ///
    /// This may tear for unaligned loads.
    #[inline]
    pub fn load_unaligned<U: RacyStorage>(&self) -> Option<U> {
        // SAFETY: Slice should always be aligned to its own type.
        unsafe { assert_unchecked(self.ptr.as_ptr().cast::<T>().is_aligned()) };
        if self.byte_length() < size_of::<U>() {
            return None;
        }

        if UNALIGNED_ACCESS_IS_OK
            || const { core::mem::align_of::<U>() <= core::mem::align_of::<T>() }
        {
            // Always properly aligned and we've checked the byte length.
            let racy = Racy::<U>::from_ptr(self.ptr.as_ptr());
            return Some(racy.load(Ordering::Unordered));
        }
        if const { core::mem::size_of::<U>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let mut scratch = MaybeUninit::<Target>::uninit();
            // SAFETY: stack addresses are non-zero.
            let dst = unsafe { NonNull::new_unchecked(scratch.as_mut_ptr().cast::<()>()) };
            // SAFETY: checked self length, dst is proper length.
            unsafe { unordered_memcpy_down_unsynchronized(self.ptr.as_ptr(), dst, size_of::<U>()) };
            // SAFETY: copy has initialised the scratch data.
            let result = unsafe { scratch.assume_init() };
            // SAFETY: type checked.
            Some(unsafe { core::mem::transmute_copy::<Target, U>(&result) })
        } else if const { core::mem::size_of::<U>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let mut scratch = MaybeUninit::<Target>::uninit();
            // SAFETY: stack addresses are non-zero.
            let dst = unsafe { NonNull::new_unchecked(scratch.as_mut_ptr().cast::<()>()) };
            // SAFETY: checked self length, dst is proper length.
            unsafe { unordered_memcpy_down_unsynchronized(self.ptr.as_ptr(), dst, size_of::<U>()) };
            // SAFETY: copy has initialised the scratch data.
            let result = unsafe { scratch.assume_init() };
            // SAFETY: type checked.
            Some(unsafe { core::mem::transmute_copy::<Target, U>(&result) })
        } else if const { core::mem::size_of::<U>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let mut scratch = MaybeUninit::<Target>::uninit();
            // SAFETY: stack addresses are non-zero.
            let dst = unsafe { NonNull::new_unchecked(scratch.as_mut_ptr().cast::<()>()) };
            // SAFETY: checked self length, dst is proper length.
            unsafe { unordered_memcpy_down_unsynchronized(self.ptr.as_ptr(), dst, size_of::<U>()) };
            // SAFETY: copy has initialised the scratch data.
            let result = unsafe { scratch.assume_init() };
            // SAFETY: type checked.
            Some(unsafe { core::mem::transmute_copy::<Target, U>(&result) })
        } else if const { core::mem::size_of::<U>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
            type Target = u64;
            let mut scratch = MaybeUninit::<Target>::uninit();
            // SAFETY: stack addresses are non-zero.
            let dst = unsafe { NonNull::new_unchecked(scratch.as_mut_ptr().cast::<()>()) };
            // SAFETY: checked self length, dst is proper length.
            unsafe { unordered_memcpy_down_unsynchronized(self.ptr.as_ptr(), dst, size_of::<U>()) };
            // SAFETY: copy has initialised the scratch data.
            let result = unsafe { scratch.assume_init() };
            // SAFETY: type checked.
            Some(unsafe { core::mem::transmute_copy::<Target, U>(&result) })
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }

    /// Store a value into the start of this slice using Unordered atomic
    /// ordering mode. The slice need not be aligned to the type.
    ///
    /// Returns None if the slice is smaller than the type.
    ///
    /// # Tearing
    ///
    /// This may tear for unaligned stores.
    #[inline]
    pub fn store_unaligned<U: RacyStorage>(&self, val: U) -> Option<()> {
        // SAFETY: Slice should always be aligned to its own type.
        unsafe { assert_unchecked(self.ptr.as_ptr().cast::<T>().is_aligned()) };
        if self.byte_length() < size_of::<U>() {
            return None;
        }

        if UNALIGNED_ACCESS_IS_OK
            || const { core::mem::align_of::<U>() <= core::mem::align_of::<T>() }
        {
            // Always properly aligned and we've checked the byte length.
            let racy = Racy::<U>::from_ptr(self.ptr.as_ptr());
            racy.store(val, Ordering::Unordered);
            return Some(());
        }
        if const { core::mem::size_of::<U>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
        }
        // SAFETY: checked self length, dst is proper length.
        let src = unsafe { NonNull::new_unchecked(&val as *const _ as *mut ()) };
        unsafe { unordered_memcpy_down_unsynchronized(src, self.ptr.as_ptr(), size_of::<U>()) };
        Some(())
    }

    /// Get the racy value at the given index.
    ///
    /// Returns None if the index is out of bounds of the slice.
    pub fn get(&self, index: usize) -> Option<Racy<'a, T>> {
        if index >= self.len() {
            None
        } else {
            // SAFETY: cannot overflow len.
            let ptr =
                RacyPtr::<T>::from_ptr(unsafe { self.ptr.as_ptr().cast::<T>().add(index).cast() });
            Some(Racy::from_ptr(ptr.as_ptr()))
        }
    }

    /// Copies all elements from `src` into `self` using a racy atomic memcpy.
    /// Note that the source slice uses the Rust memory model and this method
    /// abides by that: the source slice is only read from and no data races
    /// in the source memory are possible through the use of this function.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    pub fn copy_from_slice(&self, src: &[T]) {
        if self.len() != src.len() {
            len_mismatch_fail();
        }
        let count = self.len();
        // SAFETY: slice pointers are always non-null.
        let src = unsafe { NonNull::new_unchecked(src.as_ptr().cast_mut()) };
        let dst: NonNull<T> = self.as_ptr().as_ptr().cast();
        // SAFETY: lengths are checked to match, and racy slice cannot overlap
        // with a Rust slice.
        unsafe { unordered_copy_nonoverlapping(src, dst, count) };
    }

    /// Copies all elements from `self` into `dst`, using a racy atomic memcpy.
    /// Note that the target slice uses the Rust memory model and this method
    /// abides by that: the target slice is held exclusively during the writing
    /// and thus no data races in the target memory are possible through the
    /// use of this function.
    ///
    /// The length of `dst` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    pub fn copy_into_slice(&self, dst: &mut [T]) {
        if self.len() != dst.len() {
            len_mismatch_fail();
        }
        let count = self.len();
        let src: NonNull<T> = self.as_ptr().as_ptr().cast();
        // SAFETY: slice pointers are always non-null.
        let dst = unsafe { NonNull::new_unchecked(dst.as_ptr().cast_mut()) };
        // SAFETY: lengths are checked to match, and racy slice cannot overlap
        // with a Rust slice.
        unsafe { unordered_copy_nonoverlapping(src, dst, count) };
    }

    /// Copies all elements from `src` into `self`, using a racy memmove or
    /// memcpy as appropriate.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    pub fn copy_from_racy_slice(&self, other: &Self) {
        if self.len() != other.len() {
            len_mismatch_fail();
        }
        let count = self.len();
        // SAFETY: slice is guaranteed to point to len valid elements.
        if unsafe {
            self.ptr.as_ptr() <= other.ptr.as_ptr()
                && other.ptr.as_ptr() < self.ptr.as_ptr().byte_add(self.byte_length())
                || other.ptr.as_ptr() <= self.ptr.as_ptr()
                    && self.ptr.as_ptr() < other.ptr.as_ptr().byte_add(other.byte_length())
        } {
            // Overlapping data.
            // SAFETY: Count checked to be valid for both.
            unsafe { unordered_copy(other.as_ptr(), self.as_ptr(), count) };
        } else {
            let dst: NonNull<T> = self.as_ptr().as_ptr().cast();
            let src: NonNull<T> = other.as_ptr().as_ptr().cast();
            // SAFETY: Count checked to be valid for both.
            unsafe { unordered_copy_nonoverlapping(src, dst, count) };
        }
    }
}

pub struct RacyIter<'a, T: RacyStorage> {
    /// The pointer to the next element to return, or the past-the-end location
    /// if the iterator is empty.
    ptr: RacyPtr<T>,
    /// The non-null pointer to the past-the-end element.
    end: NonNull<()>,
    _marker: PhantomData<Racy<'a, T>>,
}

// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Send for RacyIter<'_, T> {}
// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Sync for RacyIter<'_, T> {}

impl<'a, T: RacyStorage> RacyIter<'a, T> {
    #[inline]
    const fn new(slice: RacySlice<'a, T>) -> Self {
        let len = slice.len();
        let ptr: RacyPtr<T> = slice.as_ptr();
        // SAFETY: Slice guarantees this doesn't overflow.
        let end = unsafe { ptr.as_ptr().cast::<T>().add(len).cast::<()>() };
        Self {
            ptr,
            end,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: RacyStorage> Iterator for RacyIter<'a, T> {
    type Item = Racy<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.ptr;
        let end = self.end;
        unsafe {
            if ptr.as_ptr() == end {
                return None;
            }
            // SAFETY: since we're not at the end, per the check above, moving
            // forward one keeps us inside the slice, and this is valid.
            self.ptr = RacyPtr::from_ptr(ptr.as_ptr().cast::<T>().add(1).cast::<()>());
            // SAFETY: Now that we know it wasn't empty and we've moved past
            // the first one we can give out a racy value.
            Some(Racy::from_ptr(ptr.as_ptr()))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // SAFETY: end is always geq to ptr.
        let exact = unsafe { self.end.offset_from_unsigned(self.ptr.as_ptr()) };
        (exact, Some(exact))
    }

    #[inline]
    fn count(self) -> usize {
        // SAFETY: end is always geq to ptr.
        unsafe { self.end.offset_from_unsigned(self.ptr.as_ptr()) }
    }
}

impl<'a, T: RacyStorage> DoubleEndedIterator for RacyIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ptr = self.ptr;
        let end = self.end;
        unsafe {
            if ptr.as_ptr() == end {
                return None;
            }
            // SAFETY: since we're not at the end, per the check above, moving
            // the end backward one keeps us inside the slice, and this is valid.
            self.end = end.cast::<T>().sub(1).cast::<()>();
            // SAFETY: Now that we know it wasn't empty and we've moved past
            // the first one we can give out a racy value.
            Some(Racy::from_ptr(self.end))
        }
    }
}

impl<'a, T: RacyStorage> IntoIterator for RacySlice<'a, T> {
    type Item = Racy<'a, T>;

    type IntoIter = RacyIter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[track_caller]
#[cold]
const fn len_mismatch_fail() -> ! {
    panic!("copy_from_slice: source slice length does not match destination slice length")
}

/// Opaque pointer to memory in the ECMAScript racy atomics memory model.
///
/// This is intended for unsafe usage only, where eg. the size of an EMCAScript
/// memory is stored separately from the pointer.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct RacyPtr<T: RacyStorage>(NonNull<()>, PhantomData<NonNull<UnsafeCell<T>>>);

// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Send for RacyPtr<T> {}
// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Sync for RacyPtr<T> {}

impl<T: RacyStorage> RacyPtr<T> {
    /// Creates a new racy pointer that is dangling, but non-null and
    /// well-aligned.
    #[inline(always)]
    #[must_use]
    pub const fn dangling() -> RacyPtr<T> {
        Self::from_ptr(NonNull::<T>::dangling().cast())
    }

    const fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr, PhantomData)
    }

    /// Get racy atomic pointer as a non-null pointer.
    ///
    /// # Soundness
    ///
    /// While this provides a Rust-recognisable pointer, this pointer is
    /// invalid and cannot be used to read or write any data. The only
    /// meaningful operation on the pointer is to perform possible offsetting.
    pub const fn as_ptr(self) -> NonNull<()> {
        self.0
    }

    /// Casts to a pointer of another type.
    pub const fn cast<U: RacyStorage>(self) -> RacyPtr<U> {
        RacyPtr(self.0.cast(), PhantomData)
    }
}

/// An opaque pointer to memory implementing the ECMAScript atomic memory
/// model.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Racy<'a, T: RacyStorage>(NonNull<()>, PhantomData<&'a UnsafeCell<T>>);

// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Send for Racy<'_, T> {}
// SAFETY: Racy atomics are safe to access from multiple threads.
unsafe impl<T: RacyStorage> Sync for Racy<'_, T> {}

impl<T: RacyStorage> Racy<'_, T> {
    fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr, PhantomData)
    }

    fn as_ptr(&self) -> NonNull<()> {
        self.0
    }

    /// Stores a value into the racy atomic integer if the current value is the
    /// same as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was
    /// written and containing the previous value. On success this value is
    /// guaranteed to be equal to `current`.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn compare_exchange(&self, current: T, new: T) -> Result<T, T> {
        if const { core::mem::size_of::<T>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let cur = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&current) };
            let new = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&new) };
            let old = atomic_cmp_xchg_8_seq_cst(self.as_ptr(), cur, new);
            // SAFETY: type checked.
            let old = unsafe { core::mem::transmute_copy::<Target, T>(&old) };
            if old == current { Ok(old) } else { Err(old) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let cur = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&current) };
            let new = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&new) };
            let old = atomic_cmp_xchg_16_seq_cst(self.as_ptr(), cur, new);
            // SAFETY: type checked.
            let old = unsafe { core::mem::transmute_copy::<Target, T>(&old) };
            if old == current { Ok(old) } else { Err(old) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let cur = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&current) };
            let new = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&new) };
            let old = atomic_cmp_xchg_32_seq_cst(self.as_ptr(), cur, new);
            // SAFETY: type checked.
            let old = unsafe { core::mem::transmute_copy::<Target, T>(&old) };
            if old == current { Ok(old) } else { Err(old) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u64>() } {
            type Target = u64;
            let cur = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&current) };
            let new = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&new) };
            let old = atomic_cmp_xchg_64_seq_cst(self.as_ptr(), cur, new);
            // SAFETY: type checked.
            let old = unsafe { core::mem::transmute_copy::<Target, T>(&old) };
            if old == current { Ok(old) } else { Err(old) }
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }

    /// Adds to the current value, returning the previous value.
    ///
    /// This operation wraps around on overflow.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_add(&self, val: T) -> T {
        if const { core::mem::size_of::<T>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_add_8_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_add_16_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_add_32_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
            type Target = u64;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_add_64_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }

    /// Bitwise "and" with the current value.
    ///
    /// Performs a bitwise "and" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_and(&self, val: T) -> T {
        if const { core::mem::size_of::<T>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_and_8_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_and_16_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_and_32_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
            type Target = u64;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_and_64_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }

    /// Bitwise "or" with the current value.
    ///
    /// Performs a bitwise "or" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_or(&self, val: T) -> T {
        if const { core::mem::size_of::<T>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_or_8_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_or_16_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_or_32_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
            type Target = u64;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_or_64_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }

    /// Bitwise "xor" with the current value.
    ///
    /// Performs a bitwise "xor" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_xor(&self, val: T) -> T {
        if const { core::mem::size_of::<T>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_xor_8_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_xor_16_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_xor_32_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
            type Target = u64;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_xor_64_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }

    /// Loads a value from the atomic integer.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn load(&self, order: Ordering) -> T {
        if const { core::mem::size_of::<T>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let result = if order == Ordering::SeqCst {
                atomic_load_8_seq_cst(self.as_ptr())
            } else {
                atomic_load_8_unsynchronized(self.as_ptr())
            };
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let result = if order == Ordering::SeqCst {
                atomic_load_16_seq_cst(self.as_ptr())
            } else {
                atomic_load_16_unsynchronized(self.as_ptr())
            };
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let result = if order == Ordering::SeqCst {
                atomic_load_32_seq_cst(self.as_ptr())
            } else {
                atomic_load_32_unsynchronized(self.as_ptr())
            };
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
            type Target = u64;
            let result = if order == Ordering::SeqCst {
                atomic_load_64_seq_cst(self.as_ptr())
            } else {
                atomic_load_64_unsynchronized(self.as_ptr())
            };
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }

    /// Stores a value into the atomic integer.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn store(&self, val: T, order: Ordering) {
        if const { core::mem::size_of::<T>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            if order == Ordering::SeqCst {
                atomic_store_8_seq_cst(self.as_ptr(), val)
            } else {
                atomic_store_8_unsynchronized(self.as_ptr(), val)
            }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            if order == Ordering::SeqCst {
                atomic_store_16_seq_cst(self.as_ptr(), val)
            } else {
                atomic_store_16_unsynchronized(self.as_ptr(), val)
            }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            if order == Ordering::SeqCst {
                atomic_store_32_seq_cst(self.as_ptr(), val)
            } else {
                atomic_store_32_unsynchronized(self.as_ptr(), val)
            }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
            type Target = u64;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            if order == Ordering::SeqCst {
                atomic_store_64_seq_cst(self.as_ptr(), val)
            } else {
                atomic_store_64_unsynchronized(self.as_ptr(), val)
            }
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }

    /// Stores a value into the atomic integer, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn swap(&self, val: T) -> T {
        if const { core::mem::size_of::<T>() == core::mem::size_of::<u8>() } {
            type Target = u8;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_exchange_8_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u16>() } {
            type Target = u16;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_exchange_16_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u32>() } {
            type Target = u32;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_exchange_32_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else if const { core::mem::size_of::<T>() == core::mem::size_of::<u64>() } {
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Cannot use this atomic operation on a 32-bit architecture");
            }
            type Target = u64;
            let val = // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<T, Target>(&val) };
            let result = atomic_exchange_64_seq_cst(self.as_ptr(), val);
            // SAFETY: type checked.
            unsafe { core::mem::transmute_copy::<Target, T>(&result) }
        } else {
            panic!("Unsupported T: RacyStorage");
        }
    }
}

/// Copies `count` elements from `src` to `dst`. The source and destination
/// must *not* overlap.
///
/// For regions of memory which might overlap, use [`unordered_copy`] instead.
///
/// `unordered_copy_nonoverlapping` is semantically equivalent to C's
/// [`memcpy`], but with the source and destination arguments swapped, and with
/// data races allowed.
///
/// [`memcpy`]: https://en.cppreference.com/w/c/string/byte/memcpy
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads of `count` elements.
///
/// * `dst` must be [valid] for writes of `count` elements.
///
/// * Both `src` and `dst` must be properly aligned.
///
/// * The region of memory beginning at `src` with a size of `count` elements
///   must *not* overlap with the region of memory beginning at `dst` with the
///   same size.
///
/// [valid]: https://doc.rust-lang.org/stable/core/ptr/#safety
#[inline]
unsafe fn unordered_copy_nonoverlapping<T: RacyStorage>(
    src: NonNull<T>,
    dst: NonNull<T>,
    count: usize,
) {
    let count = count * size_of::<T>();
    unsafe { unordered_memcpy_down_unsynchronized(src.cast(), dst.cast(), count) };
}

/// Copies `count` elements from `src` to `dst`. The source and destination may
/// overlap.
///
/// If the source and destination will *never* overlap,
/// [`unordered_copy_nonoverlapping`] can be used instead.
///
/// `unordered_copy` is semantically equivalent to C's [`memmove`], but with
/// the source and destination arguments swapped, and with data races allowed.
/// Copying takes place as if the elements were copied from `src` to a
/// temporary array and then copied from the array to `dst`.
///
/// [`memmove`]: https://en.cppreference.com/w/c/string/byte/memmove
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads of `count` elements, OR it must be an
///   opaque racy handle into a racy slice of `count` elements.
///
/// * `dst` must be [valid] for writes of `count` elements, and must remain
///   valid even when `src` is read for `count` elements (this means if the
///   memory ranges overlap, the `dst` pointer must not be invalidated by
///   `src` reads), OR it must be an opaque racy handle into a racy slice of
///   `count` elements.
///
/// [valid]: https://doc.rust-lang.org/stable/core/ptr/#safety
#[inline]
unsafe fn unordered_copy<T: RacyStorage>(src: RacyPtr<T>, dst: RacyPtr<T>, count: usize) {
    let count = count * size_of::<T>();
    if dst.as_ptr() <= src.as_ptr() {
        unsafe { unordered_memcpy_down_unsynchronized(src.as_ptr(), dst.as_ptr(), count) };
    } else {
        unsafe { unordered_memcpy_up_unsynchronized(src.as_ptr(), dst.as_ptr(), count) };
    }
}
