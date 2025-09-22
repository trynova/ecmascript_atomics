// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![no_main]
use core::ptr::NonNull;
use ecmascript_atomics::{
    BLOCK_SIZE, WORD_SIZE, WORDS_IN_BLOCK, atomic_add_8_seq_cst, atomic_add_16_seq_cst,
    atomic_add_32_seq_cst, atomic_add_64_seq_cst, atomic_and_8_seq_cst, atomic_and_16_seq_cst,
    atomic_and_32_seq_cst, atomic_and_64_seq_cst, atomic_cmp_xchg_8_seq_cst,
    atomic_cmp_xchg_16_seq_cst, atomic_cmp_xchg_32_seq_cst, atomic_cmp_xchg_64_seq_cst,
    atomic_copy_block_down_unsynchronized, atomic_copy_block_up_unsynchronized,
    atomic_copy_unaligned_block_down_unsynchronized, atomic_copy_unaligned_block_up_unsynchronized,
    atomic_copy_unaligned_word_down_unsynchronized, atomic_copy_unaligned_word_up_unsynchronized,
    atomic_copy_word_unsynchronized, atomic_copy8_unsynchronized, atomic_copy16_unsynchronized,
    atomic_copy32_unsynchronized, atomic_exchange_8_seq_cst, atomic_exchange_16_seq_cst,
    atomic_exchange_32_seq_cst, atomic_exchange_64_seq_cst, atomic_load_8_seq_cst,
    atomic_load_8_unsynchronized, atomic_load_16_seq_cst, atomic_load_16_unsynchronized,
    atomic_load_32_seq_cst, atomic_load_32_unsynchronized, atomic_load_64_seq_cst,
    atomic_load_64_unsynchronized, atomic_or_8_seq_cst, atomic_or_16_seq_cst, atomic_or_32_seq_cst,
    atomic_or_64_seq_cst, atomic_store_8_seq_cst, atomic_store_8_unsynchronized,
    atomic_store_16_seq_cst, atomic_store_16_unsynchronized, atomic_store_32_seq_cst,
    atomic_store_32_unsynchronized, atomic_store_64_seq_cst, atomic_store_64_unsynchronized,
    atomic_xor_8_seq_cst, atomic_xor_16_seq_cst, atomic_xor_32_seq_cst, atomic_xor_64_seq_cst,
};
use std::{
    hint::assert_unchecked,
    ops::{BitAnd, BitOr, BitXor},
};

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Clone, Copy, Debug)]
enum LoadKind {
    U8,
    U16,
    U32,
    U64,
}

#[derive(Arbitrary, Clone, Copy, Debug)]
enum StoreKind {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

#[derive(Arbitrary, Clone, Copy, Debug)]
enum CopyAlignedKind {
    U8,
    U16,
    U32,
    Usize,
    Block,
}

#[derive(Arbitrary, Clone, Copy, Debug)]
enum CopyUnalignedKind {
    Usize,
    Block,
}

#[derive(Arbitrary, Clone, Copy, Debug)]
enum FetchOp {
    Add,
    And,
    Or,
    Xor,
}

#[derive(Arbitrary, Clone, Copy, Debug)]
enum CopyDirection {
    Up,
    Down,
}

#[derive(Arbitrary, Debug)]
enum AtomicsOp {
    AtomicLoad {
        kind: LoadKind,
        offset: u8,
    },
    AtomicStore {
        kind: StoreKind,
        offset: u8,
    },
    UnorderedLoad {
        kind: LoadKind,
        offset: u8,
    },
    UnorderedStore {
        kind: StoreKind,
        offset: u8,
    },
    CompareExchange {
        kind: StoreKind,
        offset: u8,
    },
    Exchange {
        kind: StoreKind,
        offset: u8,
    },
    FetchOp {
        kind: StoreKind,
        op: FetchOp,
        offset: u8,
    },
    CopyAligned {
        kind: CopyAlignedKind,
        direction: CopyDirection,
        src_offset: u8,
        dst_offset: u8,
    },
    CopyUnaligned {
        kind: CopyUnalignedKind,
        direction: CopyDirection,
        src_offset: u8,
        dst_offset: u8,
    },
}

fn get_t<T: Copy + Sized>(rust_mem: &[u8], offset: usize) -> T {
    // SAFETY: Checked in execute_ops.
    unsafe { assert_unchecked(rust_mem.len() == ARENA_SIZE) };
    let byte_offset = offset * size_of::<T>();
    // SAFETY: T is Copy, so it's safe to make copies of it like this.
    let (head, body, tail) = unsafe { rust_mem[byte_offset..].align_to::<T>() };
    assert!(head.is_empty() && tail.is_empty());
    body[0]
}

fn get_t_mut<T: Copy + Sized>(rust_mem: &mut [u8], offset: usize) -> &mut T {
    // SAFETY: Checked in execute_ops.
    unsafe { assert_unchecked(rust_mem.len() == ARENA_SIZE) };
    let byte_offset = offset * size_of::<T>();
    // SAFETY: T is Copy, so it's safe to make copies of it like this.
    let (head, body, tail) = unsafe { rust_mem[byte_offset..].align_to_mut::<T>() };
    assert!(head.is_empty() && tail.is_empty());
    &mut body[0]
}

fn execute_ops(rust_mem: &mut [u8], ecmascript_mem: NonNull<()>, ops: &[AtomicsOp]) {
    assert_eq!(rust_mem.len(), ARENA_SIZE);
    assert!(rust_mem.as_ptr().cast::<usize>().is_aligned());
    assert!(ecmascript_mem.cast::<usize>().is_aligned());
    for op in ops {
        match op {
            AtomicsOp::AtomicLoad { kind, offset } => {
                let offset = *offset as usize;
                match kind {
                    LoadKind::U8 => {
                        let rust_val = get_t::<u8>(rust_mem, offset);
                        let byte_offset = offset * size_of::<u8>();
                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_val =
                            atomic_load_8_seq_cst(unsafe { ecmascript_mem.byte_add(byte_offset) });
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U16 => {
                        let rust_val = get_t::<u16>(rust_mem, offset);
                        let byte_offset = offset * size_of::<u16>();

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u16>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_load_16_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U32 => {
                        let rust_val = get_t::<u32>(rust_mem, offset);
                        let byte_offset = offset * size_of::<u32>();

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u32>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_load_32_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U64 => {
                        let rust_val = get_t::<u64>(rust_mem, offset);
                        let byte_offset = offset * size_of::<u64>();

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u64>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_load_64_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val)
                    }
                }
            }
            AtomicsOp::AtomicStore { kind, offset } => {
                let offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        let byte_offset = offset * size_of::<u8>();
                        rust_mem[byte_offset] = val;
                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        atomic_store_8_seq_cst(ecmascript_dst, val);
                        let ecmascript_val = atomic_load_8_seq_cst(ecmascript_dst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U16(val) => {
                        let byte_offset = offset * size_of::<u16>();
                        // SAFETY: u8s are transmutable to u16.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u16>() };
                        assert!(head.is_empty() && tail.is_empty());
                        body[0] = val;

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u16>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        atomic_store_16_seq_cst(ecmascript_dst, val);
                        let ecmascript_val = atomic_load_16_seq_cst(ecmascript_dst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U32(val) => {
                        let byte_offset = offset * size_of::<u32>();
                        // SAFETY: u8s are transmutable to u32.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u32>() };
                        assert!(head.is_empty() && tail.is_empty());
                        body[0] = val;

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u32>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        atomic_store_32_seq_cst(ecmascript_dst, val);
                        let ecmascript_val = atomic_load_32_seq_cst(ecmascript_dst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U64(val) => {
                        let byte_offset = offset * size_of::<u64>();
                        // SAFETY: u8s are transmutable to u64.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u64>() };
                        assert!(head.is_empty() && tail.is_empty());
                        body[0] = val;

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u64>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        atomic_store_64_seq_cst(ecmascript_dst, val);
                        let ecmascript_val = atomic_load_64_seq_cst(ecmascript_dst);
                        assert_eq!(ecmascript_val, val);
                    }
                }
            }
            AtomicsOp::UnorderedLoad { kind, offset } => {
                let byte_offset = *offset as usize;
                match kind {
                    LoadKind::U8 => {
                        let rust_val = rust_mem[byte_offset];
                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_val = atomic_load_8_unsynchronized(unsafe {
                            ecmascript_mem.byte_add(byte_offset)
                        });
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U16 => {
                        let bytes = rust_mem[byte_offset..].first_chunk::<2>().unwrap();
                        let rust_val = u16::from_ne_bytes(*bytes);

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_load_16_unsynchronized(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U32 => {
                        let bytes = rust_mem[byte_offset..].first_chunk::<4>().unwrap();
                        let rust_val = u32::from_ne_bytes(*bytes);

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_load_32_unsynchronized(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U64 => {
                        let bytes = rust_mem[byte_offset..].first_chunk::<8>().unwrap();
                        let rust_val = u64::from_ne_bytes(*bytes);

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_load_64_unsynchronized(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val)
                    }
                }
            }
            AtomicsOp::UnorderedStore { kind, offset } => {
                let byte_offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        rust_mem[byte_offset] = val;
                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        atomic_store_8_unsynchronized(ecmascript_dst, val);
                        let ecmascript_val = atomic_load_8_unsynchronized(ecmascript_dst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U16(val) => {
                        let bytes = rust_mem[byte_offset..].first_chunk_mut::<2>().unwrap();
                        *bytes = u16::to_ne_bytes(val);

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        // SAFETY: checked offset and alignment.
                        atomic_store_16_unsynchronized(ecmascript_dst, val);
                        let ecmascript_val = atomic_load_16_unsynchronized(ecmascript_dst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U32(val) => {
                        let bytes = rust_mem[byte_offset..].first_chunk_mut::<4>().unwrap();
                        *bytes = u32::to_ne_bytes(val);

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        // SAFETY: checked offset and alignment.
                        atomic_store_32_unsynchronized(ecmascript_dst, val);
                        let ecmascript_val = atomic_load_32_unsynchronized(ecmascript_dst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U64(val) => {
                        let bytes = rust_mem[byte_offset..].first_chunk_mut::<8>().unwrap();
                        *bytes = u64::to_ne_bytes(val);

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        // SAFETY: checked offset and alignment.
                        atomic_store_64_unsynchronized(ecmascript_dst, val);
                        let ecmascript_val = atomic_load_64_unsynchronized(ecmascript_dst);
                        assert_eq!(ecmascript_val, val);
                    }
                }
            }
            AtomicsOp::CompareExchange { kind, offset } => {
                let offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        let byte_offset = offset * size_of::<u8>();
                        let guess = val.rotate_left(4);
                        let rust_val = rust_mem[byte_offset];
                        if rust_val == guess {
                            rust_mem[byte_offset] = val;
                        }
                        if rust_mem[byte_offset] == rust_val {
                            rust_mem[byte_offset] = val;
                        }

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        let ecmascript_val = atomic_cmp_xchg_8_seq_cst(ecmascript_dst, guess, val);
                        atomic_cmp_xchg_8_seq_cst(ecmascript_dst, ecmascript_val, val);
                        let final_ecmascript_val = atomic_load_8_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(final_ecmascript_val, val);
                    }
                    StoreKind::U16(val) => {
                        let byte_offset = offset * size_of::<u16>();
                        let guess = val.rotate_left(4);
                        // SAFETY: u8s are transmutable to u16.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u16>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let rust_val = body[0];
                        if rust_val == guess {
                            body[0] = val;
                        }
                        if body[0] == rust_val {
                            body[0] = val;
                        }

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u16>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_cmp_xchg_16_seq_cst(ecmascript_dst, guess, val);
                        atomic_cmp_xchg_16_seq_cst(ecmascript_dst, ecmascript_val, val);
                        let final_ecmascript_val = atomic_load_16_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(final_ecmascript_val, val);
                    }
                    StoreKind::U32(val) => {
                        let byte_offset = offset * size_of::<u32>();
                        let guess = val.rotate_left(4);
                        // SAFETY: u8s are transmutable to u32.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u32>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let rust_val = body[0];
                        if rust_val == guess {
                            body[0] = val;
                        }
                        if body[0] == rust_val {
                            body[0] = val;
                        }

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u32>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_cmp_xchg_32_seq_cst(ecmascript_dst, guess, val);
                        atomic_cmp_xchg_32_seq_cst(ecmascript_dst, ecmascript_val, val);
                        let final_ecmascript_val = atomic_load_32_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(final_ecmascript_val, val);
                    }
                    StoreKind::U64(val) => {
                        let byte_offset = offset * size_of::<u64>();
                        let guess = val.rotate_left(4);
                        // SAFETY: u8s are transmutable to u64.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u64>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let rust_val = body[0];
                        if rust_val == guess {
                            body[0] = val;
                        }
                        if body[0] == rust_val {
                            body[0] = val;
                        }

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u64>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = atomic_cmp_xchg_64_seq_cst(ecmascript_dst, guess, val);
                        atomic_cmp_xchg_64_seq_cst(ecmascript_dst, ecmascript_val, val);
                        let final_ecmascript_val = atomic_load_64_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(final_ecmascript_val, val);
                    }
                }
            }
            AtomicsOp::Exchange { kind, offset } => {
                let offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        let rust_val = core::mem::replace(get_t_mut::<u8>(rust_mem, offset), val);

                        let byte_offset = offset * size_of::<u8>();
                        assert!(byte_offset + size_of::<u8>() < ARENA_SIZE);
                        let dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        let ecmascript_val = atomic_exchange_8_seq_cst(dst, val);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    StoreKind::U16(val) => {
                        let rust_val = core::mem::replace(get_t_mut::<u16>(rust_mem, offset), val);

                        let byte_offset = offset * size_of::<u16>();
                        assert!(byte_offset + size_of::<u16>() < ARENA_SIZE);
                        let dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        let ecmascript_val = atomic_exchange_16_seq_cst(dst, val);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    StoreKind::U32(val) => {
                        let rust_val = core::mem::replace(get_t_mut::<u32>(rust_mem, offset), val);

                        let byte_offset = offset * size_of::<u32>();
                        assert!(byte_offset + size_of::<u32>() < ARENA_SIZE);
                        let dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        let ecmascript_val = atomic_exchange_32_seq_cst(dst, val);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    StoreKind::U64(val) => {
                        let rust_val = core::mem::replace(get_t_mut::<u64>(rust_mem, offset), val);

                        let byte_offset = offset * size_of::<u64>();
                        assert!(byte_offset + size_of::<u64>() < ARENA_SIZE);
                        let dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        let ecmascript_val = atomic_exchange_64_seq_cst(dst, val);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                }
            }
            AtomicsOp::FetchOp { kind, op, offset } => {
                let offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        let byte_offset = offset * size_of::<u8>();
                        let rust_val = rust_mem[byte_offset];
                        let rust_result: u8;
                        let ecmascript_op = match op {
                            FetchOp::Add => {
                                rust_result = rust_val.wrapping_add(val);
                                atomic_add_8_seq_cst
                            }
                            FetchOp::And => {
                                rust_result = rust_val.bitand(val);
                                atomic_and_8_seq_cst
                            }
                            FetchOp::Or => {
                                rust_result = rust_val.bitor(val);
                                atomic_or_8_seq_cst
                            }
                            FetchOp::Xor => {
                                rust_result = rust_val.bitxor(val);
                                atomic_xor_8_seq_cst
                            }
                        };
                        rust_mem[byte_offset] = rust_result;

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        let ecmascript_val = ecmascript_op(ecmascript_dst, val);
                        let ecmascript_result = atomic_load_8_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(ecmascript_result, rust_result);
                    }
                    StoreKind::U16(val) => {
                        let byte_offset = offset * size_of::<u16>();
                        // SAFETY: u8s are transmutable to u16.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u16>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let rust_val = body[0];
                        let rust_result: u16;
                        let ecmascript_op = match op {
                            FetchOp::Add => {
                                rust_result = rust_val.wrapping_add(val);
                                atomic_add_16_seq_cst
                            }
                            FetchOp::And => {
                                rust_result = rust_val.bitand(val);
                                atomic_and_16_seq_cst
                            }
                            FetchOp::Or => {
                                rust_result = rust_val.bitor(val);
                                atomic_or_16_seq_cst
                            }
                            FetchOp::Xor => {
                                rust_result = rust_val.bitxor(val);
                                atomic_xor_16_seq_cst
                            }
                        };
                        body[0] = rust_result;

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u16>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = ecmascript_op(ecmascript_dst, val);
                        let ecmascript_result = atomic_load_16_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(ecmascript_result, rust_result);
                    }
                    StoreKind::U32(val) => {
                        let byte_offset = offset * size_of::<u32>();
                        // SAFETY: u8s are transmutable to u32.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u32>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let rust_val = body[0];
                        let rust_result: u32;
                        let ecmascript_op = match op {
                            FetchOp::Add => {
                                rust_result = rust_val.wrapping_add(val);
                                atomic_add_32_seq_cst
                            }
                            FetchOp::And => {
                                rust_result = rust_val.bitand(val);
                                atomic_and_32_seq_cst
                            }
                            FetchOp::Or => {
                                rust_result = rust_val.bitor(val);
                                atomic_or_32_seq_cst
                            }
                            FetchOp::Xor => {
                                rust_result = rust_val.bitxor(val);
                                atomic_xor_32_seq_cst
                            }
                        };
                        body[0] = rust_result;

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u32>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = ecmascript_op(ecmascript_dst, val);
                        let ecmascript_result = atomic_load_32_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(ecmascript_result, rust_result);
                    }
                    StoreKind::U64(val) => {
                        let byte_offset = offset * size_of::<u64>();
                        // SAFETY: u8s are transmutable to u64.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u64>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let rust_val = body[0];
                        let rust_result: u64;
                        let ecmascript_op = match op {
                            FetchOp::Add => {
                                rust_result = rust_val.wrapping_add(val);
                                atomic_add_64_seq_cst
                            }
                            FetchOp::And => {
                                rust_result = rust_val.bitand(val);
                                atomic_and_64_seq_cst
                            }
                            FetchOp::Or => {
                                rust_result = rust_val.bitor(val);
                                atomic_or_64_seq_cst
                            }
                            FetchOp::Xor => {
                                rust_result = rust_val.bitxor(val);
                                atomic_xor_64_seq_cst
                            }
                        };
                        body[0] = rust_result;

                        assert!(byte_offset < ARENA_SIZE);
                        // SAFETY: checked offset.
                        let ecmascript_dst = unsafe { ecmascript_mem.byte_add(byte_offset) };
                        assert!(ecmascript_dst.cast::<u64>().is_aligned());
                        // SAFETY: checked offset and alignment.
                        let ecmascript_val = ecmascript_op(ecmascript_dst, val);
                        let ecmascript_result = atomic_load_64_seq_cst(ecmascript_dst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(ecmascript_result, rust_result);
                    }
                }
            }
            AtomicsOp::CopyAligned {
                kind,
                direction,
                src_offset,
                dst_offset,
            } => {
                let src_offset = *src_offset as usize;
                let dst_offset = *dst_offset as usize;
                match kind {
                    CopyAlignedKind::U8 => {
                        *get_t_mut(rust_mem, dst_offset) = get_t::<u8>(rust_mem, src_offset);
                        let rust_val = get_t::<u8>(rust_mem, dst_offset);

                        let src_byte_offset = src_offset;
                        let dst_byte_offset = dst_offset;
                        assert!(
                            src_byte_offset + size_of::<u8>() < ARENA_SIZE
                                && dst_byte_offset + size_of::<u8>() < ARENA_SIZE
                        );
                        let src = unsafe { ecmascript_mem.byte_add(src_byte_offset) };
                        let dst = unsafe { ecmascript_mem.byte_add(dst_byte_offset) };
                        atomic_copy8_unsynchronized(src, dst);
                        let ecmascript_val = atomic_load_8_unsynchronized(dst);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    CopyAlignedKind::U16 => {
                        *get_t_mut(rust_mem, dst_offset) = get_t::<u16>(rust_mem, src_offset);
                        let rust_val = get_t::<u16>(rust_mem, dst_offset);

                        let src_byte_offset = src_offset * size_of::<u16>();
                        let dst_byte_offset = dst_offset * size_of::<u16>();
                        assert!(
                            src_byte_offset + size_of::<u16>() < ARENA_SIZE
                                && dst_byte_offset + size_of::<u16>() < ARENA_SIZE
                        );
                        let src = unsafe { ecmascript_mem.byte_add(src_byte_offset) };
                        let dst = unsafe { ecmascript_mem.byte_add(dst_byte_offset) };
                        atomic_copy16_unsynchronized(src, dst);
                        let ecmascript_val = atomic_load_16_unsynchronized(dst);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    CopyAlignedKind::U32 => {
                        *get_t_mut(rust_mem, dst_offset) = get_t::<u32>(rust_mem, src_offset);
                        let rust_val = get_t::<u32>(rust_mem, dst_offset);

                        let src_byte_offset = src_offset * size_of::<u32>();
                        let dst_byte_offset = dst_offset * size_of::<u32>();
                        assert!(
                            src_byte_offset + size_of::<u32>() < ARENA_SIZE
                                && dst_byte_offset + size_of::<u32>() < ARENA_SIZE
                        );
                        let src = unsafe { ecmascript_mem.byte_add(src_byte_offset) };
                        let dst = unsafe { ecmascript_mem.byte_add(dst_byte_offset) };
                        atomic_copy32_unsynchronized(src, dst);
                        let ecmascript_val = atomic_load_32_unsynchronized(dst);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    CopyAlignedKind::Usize => {
                        *get_t_mut(rust_mem, dst_offset) = get_t::<usize>(rust_mem, src_offset);
                        let rust_val = get_t::<usize>(rust_mem, dst_offset);

                        let src_byte_offset = src_offset * size_of::<usize>();
                        let dst_byte_offset = dst_offset * size_of::<usize>();
                        assert!(
                            src_byte_offset + size_of::<usize>() < ARENA_SIZE
                                && dst_byte_offset + size_of::<usize>() < ARENA_SIZE
                        );
                        let src = unsafe { ecmascript_mem.byte_add(src_byte_offset) };
                        let dst = unsafe { ecmascript_mem.byte_add(dst_byte_offset) };
                        atomic_copy_word_unsynchronized(src, dst);
                        #[cfg(target_pointer_width = "32")]
                        let ecmascript_val = atomic_load_32_unsynchronized(dst) as usize;
                        #[cfg(target_pointer_width = "64")]
                        let ecmascript_val = atomic_load_64_unsynchronized(dst) as usize;
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    CopyAlignedKind::Block => {
                        let ecmascript_op = match direction {
                            CopyDirection::Up => {
                                for index in (0..WORDS_IN_BLOCK).rev() {
                                    *get_t_mut(rust_mem, dst_offset + index) =
                                        get_t::<usize>(rust_mem, src_offset + index);
                                }
                                atomic_copy_block_up_unsynchronized
                            }
                            CopyDirection::Down => {
                                for index in 0..WORDS_IN_BLOCK {
                                    *get_t_mut(rust_mem, dst_offset + index) =
                                        get_t::<usize>(rust_mem, src_offset + index);
                                }
                                atomic_copy_block_down_unsynchronized
                            }
                        };

                        let src_byte_offset = src_offset * size_of::<usize>();
                        let dst_byte_offset = dst_offset * size_of::<usize>();
                        assert!(
                            src_byte_offset + BLOCK_SIZE < ARENA_SIZE
                                && dst_byte_offset + BLOCK_SIZE < ARENA_SIZE
                        );
                        let src = unsafe { ecmascript_mem.byte_add(src_byte_offset) };
                        let dst = unsafe { ecmascript_mem.byte_add(dst_byte_offset) };
                        ecmascript_op(src, dst);
                        for index in 0..WORDS_IN_BLOCK {
                            let rust_val = get_t::<usize>(rust_mem, dst_offset + index);
                            let dst = unsafe { dst.byte_add(size_of::<usize>() * index) };
                            #[cfg(target_pointer_width = "32")]
                            let ecmascript_val = atomic_load_32_unsynchronized(dst) as usize;
                            #[cfg(target_pointer_width = "64")]
                            let ecmascript_val = atomic_load_64_unsynchronized(dst) as usize;
                            assert_eq!(rust_val, ecmascript_val);
                        }
                    }
                }
            }
            AtomicsOp::CopyUnaligned {
                kind,
                direction,
                src_offset,
                dst_offset,
            } => {
                let src_byte_offset = *src_offset as usize;
                let dst_byte_offset = *dst_offset as usize;
                match kind {
                    CopyUnalignedKind::Usize => {
                        let ecmascript_op = match direction {
                            CopyDirection::Up => {
                                for index in (0..WORD_SIZE).rev() {
                                    *get_t_mut(rust_mem, dst_byte_offset + index) =
                                        get_t::<u8>(rust_mem, src_byte_offset + index);
                                }
                                atomic_copy_unaligned_word_up_unsynchronized
                            }
                            CopyDirection::Down => {
                                for index in 0..WORD_SIZE {
                                    *get_t_mut(rust_mem, dst_byte_offset + index) =
                                        get_t::<u8>(rust_mem, src_byte_offset + index);
                                }
                                atomic_copy_unaligned_word_down_unsynchronized
                            }
                        };

                        assert!(
                            src_byte_offset + WORD_SIZE < ARENA_SIZE
                                && dst_byte_offset + WORD_SIZE < ARENA_SIZE
                        );
                        let src = unsafe { ecmascript_mem.byte_add(src_byte_offset) };
                        let dst = unsafe { ecmascript_mem.byte_add(dst_byte_offset) };
                        ecmascript_op(src, dst);
                        for index in 0..WORD_SIZE {
                            let rust_val = get_t::<u8>(rust_mem, dst_byte_offset + index);
                            let dst = unsafe { dst.byte_add(index) };
                            let ecmascript_val = atomic_load_8_unsynchronized(dst);
                            assert_eq!(rust_val, ecmascript_val);
                        }
                    }
                    CopyUnalignedKind::Block => {
                        let ecmascript_op = match direction {
                            CopyDirection::Up => {
                                for index in (0..BLOCK_SIZE).rev() {
                                    *get_t_mut(rust_mem, dst_byte_offset + index) =
                                        get_t::<u8>(rust_mem, src_byte_offset + index);
                                }
                                atomic_copy_unaligned_block_up_unsynchronized
                            }
                            CopyDirection::Down => {
                                for index in 0..BLOCK_SIZE {
                                    *get_t_mut(rust_mem, dst_byte_offset + index) =
                                        get_t::<u8>(rust_mem, src_byte_offset + index);
                                }
                                atomic_copy_unaligned_block_down_unsynchronized
                            }
                        };

                        assert!(
                            src_byte_offset + BLOCK_SIZE < ARENA_SIZE
                                && dst_byte_offset + BLOCK_SIZE < ARENA_SIZE
                        );
                        let src = unsafe { ecmascript_mem.byte_add(src_byte_offset) };
                        let dst = unsafe { ecmascript_mem.byte_add(dst_byte_offset) };
                        ecmascript_op(src, dst);
                        for index in 0..BLOCK_SIZE {
                            let rust_val = get_t::<u8>(rust_mem, dst_byte_offset + index);
                            let dst = unsafe { dst.byte_add(index) };
                            let ecmascript_val = atomic_load_8_unsynchronized(dst);
                            assert_eq!(rust_val, ecmascript_val);
                        }
                    }
                }
            }
        }
    }
}

/// Size of allocated memory slab, large enough for any u8 offset to still have
/// enough memory behind it to read or write a full block.
const ARENA_SIZE: usize = (u8::MAX as usize) * size_of::<usize>() * 8;

#[derive(Arbitrary, Debug)]
struct AtomicsFuzzInput {
    ops: Vec<AtomicsOp>,
}

fuzz_target!(|input: AtomicsFuzzInput| {
    let mut rust_dst = Box::new([0usize; (u8::MAX as usize) * 8]);
    // SAFETY: can transmute usize to u8s.
    let (head, rust_dst, tail) = unsafe { rust_dst.align_to_mut::<u8>() };
    assert!(head.is_empty() && rust_dst.len() == ARENA_SIZE && tail.is_empty());
    let mut ecmascript_dst = memmap2::MmapMut::map_anon(ARENA_SIZE).expect("Failed to mmap");
    assert!(ecmascript_dst.as_ptr().cast::<u64>().is_aligned());
    assert_eq!(ecmascript_dst.len(), ARENA_SIZE);
    ecmascript_dst.fill(0);
    let ecmascript_dst: memmap2::MmapRaw = ecmascript_dst.into();

    execute_ops(
        rust_dst,
        NonNull::new(ecmascript_dst.as_mut_ptr()).unwrap().cast(),
        &input.ops,
    );
});
