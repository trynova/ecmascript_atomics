// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Copyright of the originating code is owned by Firefox authors and Mozilla,
// modifications by Aapo Alasuutari.

use core::{hint::assert_unchecked, num::NonZeroUsize, ptr::NonNull};

use crate::generated::{
    BLOCK_SIZE, WORD_SIZE, atomic_copy_block_down_unsynchronized,
    atomic_copy_block_up_unsynchronized, atomic_copy_unaligned_block_down_unsynchronized,
    atomic_copy_unaligned_block_up_unsynchronized, atomic_copy_unaligned_word_down_unsynchronized,
    atomic_copy_unaligned_word_up_unsynchronized, atomic_copy_word_unsynchronized,
    atomic_copy8_unsynchronized, atomic_copy16_unsynchronized, atomic_copy32_unsynchronized,
};

/// On x86, unaligned accesses work fine.
pub(crate) const UNALIGNED_ACCESS_IS_OK: bool =
    cfg!(any(target_arch = "x86", target_arch = "x86_64"));

fn can_copy_aligned<const ALIGNMENT: usize>(
    src: NonNull<()>,
    dst: NonNull<()>,
    lim: NonNull<()>,
) -> bool {
    const { assert!(ALIGNMENT.is_power_of_two() && ALIGNMENT > 0) };
    ((src.addr().get() | dst.addr().get() | lim.addr().get()) & (ALIGNMENT - 1)) == 0
}

fn can_align_to<const ALIGNMENT: usize>(src: NonNull<()>, dst: NonNull<()>) -> bool {
    const { assert!(ALIGNMENT.is_power_of_two() && ALIGNMENT > 0) };
    ((src.addr().get() ^ dst.addr().get()) & (ALIGNMENT - 1)) == 0
}

const BLOCK_MASK: usize = BLOCK_SIZE - 1;
const WORD_MASK: usize = WORD_SIZE - 1;

#[inline(always)]
pub(crate) unsafe fn unordered_memcpy_down_unsynchronized(
    mut src: NonNull<()>,
    mut dst: NonNull<()>,
    count: usize,
) {
    // SAFETY: Caller guaranteed.
    let lim = unsafe { src.byte_add(count) };

    // Set up bulk copying.  The cases are ordered the way they are on the
    // assumption that if we can achieve aligned copies even with a little
    // preprocessing then that is better than unaligned copying on a platform
    // that supports it.

    if count >= WORD_SIZE {
        let copy_block: fn(NonNull<()>, NonNull<()>);
        let copy_word: fn(NonNull<()>, NonNull<()>);

        if can_align_to::<WORD_SIZE>(src, dst) {
            // SAFETY: Cannot overflow because count >= WORD_SIZE.
            let cutoff = unsafe { src.byte_add(src.cast::<u8>().align_offset(WORD_SIZE)) };
            debug_assert!(cutoff <= lim); // because count >= WORD_SIZE

            // Copy initial bytes to align to word size.
            (src, dst) = atomic_copy_down_no_tear_if_aligned_unsynchronized(src, dst, cutoff);

            copy_block = atomic_copy_block_down_unsynchronized;
            copy_word = atomic_copy_word_unsynchronized;
        } else if UNALIGNED_ACCESS_IS_OK {
            copy_block = atomic_copy_block_down_unsynchronized;
            copy_word = atomic_copy_word_unsynchronized;
        } else {
            copy_block = atomic_copy_unaligned_block_down_unsynchronized;
            copy_word = atomic_copy_unaligned_word_down_unsynchronized;
        }

        // Bulk copy, first larger blocks and then individual words.

        let block_lim = unsafe { src.byte_add(lim.byte_offset_from_unsigned(src) & !BLOCK_MASK) };
        while src < block_lim {
            copy_block(src, dst);
            // SAFETY: Checked.
            unsafe {
                dst = dst.byte_add(BLOCK_SIZE);
                src = src.byte_add(BLOCK_SIZE);
            }
        }

        let word_lim = unsafe { src.byte_add(lim.byte_offset_from_unsigned(src) & !WORD_MASK) };
        while src < word_lim {
            copy_word(src, dst);
            // SAFETY: Checked.
            unsafe {
                dst = dst.byte_add(WORD_SIZE);
                src = src.byte_add(WORD_SIZE);
            }
        }
    }

    // Copy any remaining tail.

    atomic_copy_down_no_tear_if_aligned_unsynchronized(src, dst, lim);
}

#[inline(always)]
pub(crate) unsafe fn unordered_memcpy_up_unsynchronized(
    mut src: NonNull<()>,
    mut dst: NonNull<()>,
    count: usize,
) {
    let lim = src;

    // SAFETY: Caller guaranteed.
    unsafe {
        src = src.byte_add(count);
        dst = dst.byte_add(count);
    }

    // Set up bulk copying.  The cases are ordered the way they are on the
    // assumption that if we can achieve aligned copies even with a little
    // preprocessing then that is better than unaligned copying on a platform
    // that supports it.

    if count >= WORD_SIZE {
        let copy_block: fn(NonNull<()>, NonNull<()>);
        let copy_word: fn(NonNull<()>, NonNull<()>);

        if can_align_to::<WORD_SIZE>(src, dst) {
            // SAFETY: src necessary has non-zero bits beyond the WORD_MASK
            // bottom bits as `count >= WORD_SIZE` and
            // `src = src.byte_add(count)`.
            let cutoff_addr = unsafe { NonZeroUsize::new_unchecked(src.addr().get() & !WORD_MASK) };
            let cutoff = src.with_addr(cutoff_addr);
            debug_assert!(cutoff >= lim); // Because count >= WORD_SIZE

            // Copy initial bytes to align to word size.
            (src, dst) = atomic_copy_up_no_tear_if_aligned_unsynchronized(src, dst, cutoff);

            copy_block = atomic_copy_block_up_unsynchronized;
            copy_word = atomic_copy_word_unsynchronized;
        } else if UNALIGNED_ACCESS_IS_OK {
            copy_block = atomic_copy_block_up_unsynchronized;
            copy_word = atomic_copy_word_unsynchronized;
        } else {
            copy_block = atomic_copy_unaligned_block_up_unsynchronized;
            copy_word = atomic_copy_unaligned_word_up_unsynchronized;
        }

        // Bulk copy, first larger blocks and then individual words.

        let block_lim = unsafe { src.byte_sub(src.byte_offset_from_unsigned(lim) & !BLOCK_MASK) };
        while src > block_lim {
            // SAFETY: Checked.
            unsafe {
                src = src.byte_sub(BLOCK_SIZE);
                dst = dst.byte_sub(BLOCK_SIZE);
            }
            copy_block(src, dst);
        }

        let word_lim = unsafe { src.byte_sub(src.byte_offset_from_unsigned(lim) & !WORD_MASK) };
        while src > word_lim {
            // SAFETY: Checked.
            unsafe {
                src = src.byte_sub(WORD_SIZE);
                dst = dst.byte_sub(WORD_SIZE);
            }
            copy_word(src, dst);
        }
    }

    // Copy any remaining tail.

    atomic_copy_up_no_tear_if_aligned_unsynchronized(src, dst, lim);
}

/// Copy a datum smaller than `WORD_SIZE`. Prevents tearing when `src` and `dst`
/// are both aligned.
///
/// No tearing is a requirement for integer TypedArrays.
///
/// https://tc39.es/ecma262/#sec-isnotearconfiguration
/// https://tc39.es/ecma262/#sec-tear-free-aligned-reads
/// https://tc39.es/ecma262/#sec-valid-executions
#[inline(always)]
fn atomic_copy_up_no_tear_if_aligned_unsynchronized(
    mut src: NonNull<()>,
    mut dst: NonNull<()>,
    src_begin: NonNull<()>,
) -> (NonNull<()>, NonNull<()>) {
    // SAFETY: Checking caller guarantees.
    unsafe {
        assert_unchecked(src >= src_begin);
        assert_unchecked(src.byte_offset_from_unsigned(src_begin) < WORD_SIZE);
    }

    if WORD_SIZE > 4 && can_copy_aligned::<4>(src, dst, src_begin) {
        const { assert!(WORD_SIZE <= 8, "copies 32-bits at most once") };

        if src > src_begin {
            const SIZE: usize = size_of::<u32>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_sub(SIZE);
                dst = dst.byte_sub(SIZE);
            }
            atomic_copy32_unsynchronized(src, dst);
        }
    } else if can_copy_aligned::<2>(src, dst, src_begin) {
        while src > src_begin {
            const SIZE: usize = size_of::<u16>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_sub(SIZE);
                dst = dst.byte_sub(SIZE);
            }
            atomic_copy16_unsynchronized(src, dst);
        }
    } else {
        while src > src_begin {
            const SIZE: usize = size_of::<u8>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_sub(SIZE);
                dst = dst.byte_sub(SIZE);
            }
            atomic_copy8_unsynchronized(src, dst);
        }
    }
    (src, dst)
}

#[inline(always)]
pub(crate) fn atomic_copy_down_no_tear_if_aligned_unsynchronized(
    mut src: NonNull<()>,
    mut dst: NonNull<()>,
    src_end: NonNull<()>,
) -> (NonNull<()>, NonNull<()>) {
    // SAFETY: Checked by caller.
    unsafe { assert_unchecked(src <= src_end) };
    // SAFETY: Checked by caller.
    unsafe { assert_unchecked(src_end.byte_offset_from_unsigned(src) < WORD_SIZE) };

    if WORD_SIZE > 4 && can_copy_aligned::<4>(src, dst, src_end) {
        const { assert!(WORD_SIZE <= 8, "copies 32-bits at most once") };
        if src < src_end {
            atomic_copy32_unsynchronized(src, dst);
            const SIZE: usize = size_of::<u32>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_add(SIZE);
                dst = dst.byte_add(SIZE);
            }
        }
    } else if can_copy_aligned::<2>(src, dst, src_end) {
        while src < src_end {
            atomic_copy16_unsynchronized(src, dst);
            const SIZE: usize = size_of::<u16>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_add(SIZE);
                dst = dst.byte_add(SIZE);
            }
        }
    } else {
        while src < src_end {
            atomic_copy8_unsynchronized(src, dst);
            const SIZE: usize = size_of::<u8>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_add(SIZE);
                dst = dst.byte_add(SIZE);
            }
        }
    }
    (src, dst)
}
