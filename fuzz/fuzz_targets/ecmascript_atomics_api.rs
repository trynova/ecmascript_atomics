// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![no_main]
use core::ptr::NonNull;
use ecmascript_atomics::{
    Ordering, RacyAtomicSlice, RacyAtomicU8, RacyAtomicU16, RacyAtomicU32, RacyAtomicU64,
    unordered_copy, unordered_copy_nonoverlapping,
};
use std::{
    hint::assert_unchecked,
    ops::{BitAnd, BitOr, BitXor},
    sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64},
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
enum FetchOp {
    Add,
    And,
    Or,
    Xor,
}

#[derive(Arbitrary, Clone, Copy, Debug)]
enum OpOrdering {
    SeqCst,
    Unordered,
}

impl From<OpOrdering> for Ordering {
    fn from(value: OpOrdering) -> Self {
        match value {
            OpOrdering::SeqCst => Self::SeqCst,
            OpOrdering::Unordered => Self::Unordered,
        }
    }
}

impl From<Ordering> for OpOrdering {
    fn from(value: Ordering) -> Self {
        match value {
            Ordering::SeqCst => Self::SeqCst,
            Ordering::Unordered => Self::Unordered,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum AtomicsOp {
    AtomicLoad {
        kind: LoadKind,
        offset: u8,
        order: OpOrdering,
    },
    AtomicStore {
        kind: StoreKind,
        offset: u8,
        order: OpOrdering,
    },
    UnalignedLoad {
        kind: LoadKind,
        offset: u8,
    },
    UnalignedStore {
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
    Copy {
        src_offset: u16,
        dst_offset: u16,
        len: u16,
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

fn execute_ops(rust_mem: &mut [u8], ecmascript_mem: RacyAtomicSlice, ops: &[AtomicsOp]) {
    assert_eq!(rust_mem.len(), ARENA_SIZE);
    assert!(rust_mem.as_ptr().cast::<usize>().is_aligned());
    assert!(ecmascript_mem.is_aligned::<usize>());
    for op in ops {
        match op {
            AtomicsOp::AtomicLoad {
                kind,
                offset,
                order,
            } => {
                let offset = *offset as usize;
                match kind {
                    LoadKind::U8 => {
                        let rust_val = get_t::<u8>(rust_mem, offset);
                        let byte_offset = offset * size_of::<u8>();
                        let ecmascript_val = ecmascript_mem
                            .slice_from(byte_offset)
                            .as_u8()
                            .unwrap()
                            .load((*order).into());
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U16 => {
                        let rust_val = get_t::<u16>(rust_mem, offset);
                        let byte_offset = offset * size_of::<u16>();

                        let ecmascript_val = ecmascript_mem
                            .slice_from(byte_offset)
                            .as_u16()
                            .unwrap()
                            .load((*order).into());
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U32 => {
                        let rust_val = get_t::<u32>(rust_mem, offset);
                        let byte_offset = offset * size_of::<u32>();

                        let ecmascript_val = ecmascript_mem
                            .slice_from(byte_offset)
                            .as_u32()
                            .unwrap()
                            .load((*order).into());
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U64 => {
                        let rust_val = get_t::<u64>(rust_mem, offset);
                        let byte_offset = offset * size_of::<u64>();

                        let ecmascript_val = ecmascript_mem
                            .slice_from(byte_offset)
                            .as_u64()
                            .unwrap()
                            .load((*order).into());
                        assert_eq!(rust_val, ecmascript_val)
                    }
                }
            }
            AtomicsOp::AtomicStore {
                kind,
                offset,
                order,
            } => {
                let offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        let byte_offset = offset * size_of::<u8>();
                        rust_mem[byte_offset] = val;

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u8().unwrap();
                        ecmascript_mem.store(val, (*order).into());
                        let ecmascript_val = ecmascript_mem.load(Ordering::SeqCst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U16(val) => {
                        let byte_offset = offset * size_of::<u16>();
                        // SAFETY: u8s are transmutable to u16.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u16>() };
                        assert!(head.is_empty() && tail.is_empty());
                        body[0] = val;

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u16().unwrap();
                        ecmascript_mem.store(val, (*order).into());
                        let ecmascript_val = ecmascript_mem.load(Ordering::SeqCst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U32(val) => {
                        let byte_offset = offset * size_of::<u32>();
                        // SAFETY: u8s are transmutable to u32.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u32>() };
                        assert!(head.is_empty() && tail.is_empty());
                        body[0] = val;

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u32().unwrap();
                        ecmascript_mem.store(val, (*order).into());
                        let ecmascript_val = ecmascript_mem.load(Ordering::SeqCst);
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U64(val) => {
                        let byte_offset = offset * size_of::<u64>();
                        // SAFETY: u8s are transmutable to u64.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u64>() };
                        assert!(head.is_empty() && tail.is_empty());
                        body[0] = val;

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u64().unwrap();
                        ecmascript_mem.store(val, (*order).into());
                        let ecmascript_val = ecmascript_mem.load(Ordering::SeqCst);
                        assert_eq!(ecmascript_val, val);
                    }
                }
            }
            AtomicsOp::UnalignedLoad { kind, offset } => {
                let byte_offset = *offset as usize;
                match kind {
                    LoadKind::U8 => {
                        let rust_val = rust_mem[byte_offset];

                        let ecmascript_val =
                            ecmascript_mem.slice_from(byte_offset).load_u8().unwrap();
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U16 => {
                        let bytes = rust_mem[byte_offset..].first_chunk::<2>().unwrap();
                        let rust_val = u16::from_ne_bytes(*bytes);

                        let ecmascript_val =
                            ecmascript_mem.slice_from(byte_offset).load_u16().unwrap();
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U32 => {
                        let bytes = rust_mem[byte_offset..].first_chunk::<4>().unwrap();
                        let rust_val = u32::from_ne_bytes(*bytes);

                        let ecmascript_val =
                            ecmascript_mem.slice_from(byte_offset).load_u32().unwrap();
                        assert_eq!(rust_val, ecmascript_val)
                    }
                    LoadKind::U64 => {
                        let bytes = rust_mem[byte_offset..].first_chunk::<8>().unwrap();
                        let rust_val = u64::from_ne_bytes(*bytes);

                        let ecmascript_val =
                            ecmascript_mem.slice_from(byte_offset).load_u64().unwrap();
                        assert_eq!(rust_val, ecmascript_val)
                    }
                }
            }
            AtomicsOp::UnalignedStore { kind, offset } => {
                let byte_offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        rust_mem[byte_offset] = val;

                        let ecmascript_mem = ecmascript_mem.slice_from(byte_offset);
                        ecmascript_mem.store_u8(val).unwrap();
                        let ecmascript_val = ecmascript_mem.load_u8().unwrap();
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U16(val) => {
                        let bytes = rust_mem[byte_offset..].first_chunk_mut::<2>().unwrap();
                        *bytes = u16::to_ne_bytes(val);

                        let ecmascript_mem = ecmascript_mem.slice_from(byte_offset);
                        ecmascript_mem.store_u16(val).unwrap();
                        let ecmascript_val = ecmascript_mem.load_u16().unwrap();
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U32(val) => {
                        let bytes = rust_mem[byte_offset..].first_chunk_mut::<4>().unwrap();
                        *bytes = u32::to_ne_bytes(val);

                        let ecmascript_mem = ecmascript_mem.slice_from(byte_offset);
                        ecmascript_mem.store_u32(val).unwrap();
                        let ecmascript_val = ecmascript_mem.load_u32().unwrap();
                        assert_eq!(ecmascript_val, val);
                    }
                    StoreKind::U64(val) => {
                        let bytes = rust_mem[byte_offset..].first_chunk_mut::<8>().unwrap();
                        *bytes = u64::to_ne_bytes(val);

                        let ecmascript_mem = ecmascript_mem.slice_from(byte_offset);
                        ecmascript_mem.store_u64(val).unwrap();
                        let ecmascript_val = ecmascript_mem.load_u64().unwrap();
                        assert_eq!(ecmascript_val, val);
                    }
                }
            }
            AtomicsOp::CompareExchange { kind, offset } => {
                let offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        let byte_offset = offset * size_of::<u8>();
                        let guess = 0;
                        // SAFETY: alignment checked, synchronisation through &mut.
                        let rust_mem =
                            unsafe { AtomicU8::from_ptr(&mut rust_mem[byte_offset] as *mut _) };
                        let rust_val = rust_mem.compare_exchange(
                            guess,
                            val,
                            std::sync::atomic::Ordering::SeqCst,
                            std::sync::atomic::Ordering::SeqCst,
                        );
                        if let Err(current) = rust_val {
                            rust_mem
                                .compare_exchange(
                                    current,
                                    val,
                                    std::sync::atomic::Ordering::SeqCst,
                                    std::sync::atomic::Ordering::SeqCst,
                                )
                                .unwrap();
                        }
                        let final_rust_val = rust_mem.load(std::sync::atomic::Ordering::Relaxed);

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u8().unwrap();
                        let ecmascript_val = ecmascript_mem.compare_exchange(guess, val);
                        if let Err(current) = ecmascript_val {
                            ecmascript_mem.compare_exchange(current, val).unwrap();
                        }
                        let final_ecmascript_val = ecmascript_mem.load(Ordering::Unordered);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(final_ecmascript_val, val);
                        assert_eq!(final_rust_val, final_ecmascript_val);
                    }
                    StoreKind::U16(val) => {
                        let byte_offset = offset * size_of::<u16>();
                        // SAFETY: u8s are transmutable to u16.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u16>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let guess = 0;
                        // SAFETY: alignment checked, synchronisation through &mut.
                        let rust_mem = unsafe { AtomicU16::from_ptr(&mut body[0] as *mut _) };
                        let rust_val = rust_mem.compare_exchange(
                            guess,
                            val,
                            std::sync::atomic::Ordering::SeqCst,
                            std::sync::atomic::Ordering::SeqCst,
                        );
                        if let Err(current) = rust_val {
                            rust_mem
                                .compare_exchange(
                                    current,
                                    val,
                                    std::sync::atomic::Ordering::SeqCst,
                                    std::sync::atomic::Ordering::SeqCst,
                                )
                                .unwrap();
                        }
                        let final_rust_val = rust_mem.load(std::sync::atomic::Ordering::Relaxed);

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u16().unwrap();
                        let ecmascript_val = ecmascript_mem.compare_exchange(guess, val);
                        if let Err(current) = ecmascript_val {
                            ecmascript_mem.compare_exchange(current, val).unwrap();
                        }
                        let final_ecmascript_val = ecmascript_mem.load(Ordering::Unordered);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(final_ecmascript_val, val);
                        assert_eq!(final_rust_val, final_ecmascript_val);
                    }
                    StoreKind::U32(val) => {
                        let byte_offset = offset * size_of::<u32>();
                        // SAFETY: u8s are transmutable to u32.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u32>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let guess = 0;
                        // SAFETY: alignment checked, synchronisation through &mut.
                        let rust_mem = unsafe { AtomicU32::from_ptr(&mut body[0] as *mut _) };
                        let rust_val = rust_mem.compare_exchange(
                            guess,
                            val,
                            std::sync::atomic::Ordering::SeqCst,
                            std::sync::atomic::Ordering::SeqCst,
                        );
                        if let Err(current) = rust_val {
                            rust_mem
                                .compare_exchange(
                                    current,
                                    val,
                                    std::sync::atomic::Ordering::SeqCst,
                                    std::sync::atomic::Ordering::SeqCst,
                                )
                                .unwrap();
                        }
                        let final_rust_val = rust_mem.load(std::sync::atomic::Ordering::Relaxed);

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u32().unwrap();
                        let ecmascript_val = ecmascript_mem.compare_exchange(guess, val);
                        if let Err(current) = ecmascript_val {
                            ecmascript_mem.compare_exchange(current, val).unwrap();
                        }
                        let final_ecmascript_val = ecmascript_mem.load(Ordering::Unordered);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(final_ecmascript_val, val);
                        assert_eq!(final_rust_val, final_ecmascript_val);
                    }
                    StoreKind::U64(val) => {
                        let byte_offset = offset * size_of::<u64>();
                        // SAFETY: u8s are transmutable to u64.
                        let (head, body, tail) =
                            unsafe { rust_mem[byte_offset..].align_to_mut::<u64>() };
                        assert!(head.is_empty() && tail.is_empty());
                        let guess = 0;
                        // SAFETY: alignment checked, synchronisation through &mut.
                        let rust_mem = unsafe { AtomicU64::from_ptr(&mut body[0] as *mut _) };
                        let rust_val = rust_mem.compare_exchange(
                            guess,
                            val,
                            std::sync::atomic::Ordering::SeqCst,
                            std::sync::atomic::Ordering::SeqCst,
                        );
                        if let Err(current) = rust_val {
                            rust_mem
                                .compare_exchange(
                                    current,
                                    val,
                                    std::sync::atomic::Ordering::SeqCst,
                                    std::sync::atomic::Ordering::SeqCst,
                                )
                                .unwrap();
                        }
                        let final_rust_val = rust_mem.load(std::sync::atomic::Ordering::Relaxed);

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u64().unwrap();
                        let ecmascript_val = ecmascript_mem.compare_exchange(guess, val);
                        if let Err(current) = ecmascript_val {
                            ecmascript_mem.compare_exchange(current, val).unwrap();
                        }
                        let final_ecmascript_val = ecmascript_mem.load(Ordering::Unordered);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(final_ecmascript_val, val);
                        assert_eq!(final_rust_val, final_ecmascript_val);
                    }
                }
            }
            AtomicsOp::Exchange { kind, offset } => {
                let offset = *offset as usize;
                match *kind {
                    StoreKind::U8(val) => {
                        let rust_val = core::mem::replace(get_t_mut::<u8>(rust_mem, offset), val);

                        let byte_offset = offset * size_of::<u8>();
                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u8().unwrap();
                        let ecmascript_val = ecmascript_mem.swap(val);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    StoreKind::U16(val) => {
                        let rust_val = core::mem::replace(get_t_mut::<u16>(rust_mem, offset), val);

                        let byte_offset = offset * size_of::<u16>();
                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u16().unwrap();
                        let ecmascript_val = ecmascript_mem.swap(val);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    StoreKind::U32(val) => {
                        let rust_val = core::mem::replace(get_t_mut::<u32>(rust_mem, offset), val);

                        let byte_offset = offset * size_of::<u32>();
                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u32().unwrap();
                        let ecmascript_val = ecmascript_mem.swap(val);
                        assert_eq!(rust_val, ecmascript_val);
                    }
                    StoreKind::U64(val) => {
                        let rust_val = core::mem::replace(get_t_mut::<u64>(rust_mem, offset), val);

                        let byte_offset = offset * size_of::<u64>();
                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u64().unwrap();
                        let ecmascript_val = ecmascript_mem.swap(val);
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
                                RacyAtomicU8::fetch_add
                            }
                            FetchOp::And => {
                                rust_result = rust_val.bitand(val);
                                RacyAtomicU8::fetch_and
                            }
                            FetchOp::Or => {
                                rust_result = rust_val.bitor(val);
                                RacyAtomicU8::fetch_or
                            }
                            FetchOp::Xor => {
                                rust_result = rust_val.bitxor(val);
                                RacyAtomicU8::fetch_xor
                            }
                        };
                        rust_mem[byte_offset] = rust_result;

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u8().unwrap();
                        let ecmascript_val = ecmascript_op(&ecmascript_mem, val);
                        let ecmascript_result = ecmascript_mem.load(Ordering::SeqCst);
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
                                RacyAtomicU16::fetch_add
                            }
                            FetchOp::And => {
                                rust_result = rust_val.bitand(val);
                                RacyAtomicU16::fetch_and
                            }
                            FetchOp::Or => {
                                rust_result = rust_val.bitor(val);
                                RacyAtomicU16::fetch_or
                            }
                            FetchOp::Xor => {
                                rust_result = rust_val.bitxor(val);
                                RacyAtomicU16::fetch_xor
                            }
                        };
                        body[0] = rust_result;

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u16().unwrap();
                        let ecmascript_val = ecmascript_op(&ecmascript_mem, val);
                        let ecmascript_result = ecmascript_mem.load(Ordering::SeqCst);
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
                                RacyAtomicU32::fetch_add
                            }
                            FetchOp::And => {
                                rust_result = rust_val.bitand(val);
                                RacyAtomicU32::fetch_and
                            }
                            FetchOp::Or => {
                                rust_result = rust_val.bitor(val);
                                RacyAtomicU32::fetch_or
                            }
                            FetchOp::Xor => {
                                rust_result = rust_val.bitxor(val);
                                RacyAtomicU32::fetch_xor
                            }
                        };
                        body[0] = rust_result;

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u32().unwrap();
                        let ecmascript_val = ecmascript_op(&ecmascript_mem, val);
                        let ecmascript_result = ecmascript_mem.load(Ordering::SeqCst);
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
                                RacyAtomicU64::fetch_add
                            }
                            FetchOp::And => {
                                rust_result = rust_val.bitand(val);
                                RacyAtomicU64::fetch_and
                            }
                            FetchOp::Or => {
                                rust_result = rust_val.bitor(val);
                                RacyAtomicU64::fetch_or
                            }
                            FetchOp::Xor => {
                                rust_result = rust_val.bitxor(val);
                                RacyAtomicU64::fetch_xor
                            }
                        };
                        body[0] = rust_result;

                        let ecmascript_mem =
                            ecmascript_mem.slice_from(byte_offset).as_u64().unwrap();
                        let ecmascript_val = ecmascript_op(&ecmascript_mem, val);
                        let ecmascript_result = ecmascript_mem.load(Ordering::SeqCst);
                        assert_eq!(rust_val, ecmascript_val);
                        assert_eq!(ecmascript_result, rust_result);
                    }
                }
            }
            AtomicsOp::Copy {
                src_offset,
                dst_offset,
                len,
            } => {
                // Bound the byte values to the arena size. We use modulo to
                // not over-sample the ARENA_SIZE itself.
                let src_byte_offset = (*src_offset as usize) % ARENA_SIZE;
                let dst_byte_offset = (*dst_offset as usize) % ARENA_SIZE;
                let max_offset = src_byte_offset.max(dst_byte_offset);
                let len = (((*len as usize) + max_offset) % ARENA_SIZE).saturating_sub(max_offset);
                if len == 0 {
                    continue;
                }
                rust_mem.copy_within(src_byte_offset..src_byte_offset + len, dst_byte_offset);

                let ecmascript_src = ecmascript_mem.slice(src_byte_offset, src_byte_offset + len);
                let ecmascript_dst = ecmascript_mem.slice(dst_byte_offset, dst_byte_offset + len);
                if (src_byte_offset..src_byte_offset + len).contains(&dst_byte_offset)
                    || (dst_byte_offset..dst_byte_offset + len).contains(&src_byte_offset)
                {
                    // Overlapping.
                    unsafe {
                        unordered_copy(
                            ecmascript_src.as_u8().unwrap(),
                            ecmascript_dst.as_u8().unwrap(),
                            len,
                        )
                    };
                } else {
                    unsafe {
                        unordered_copy_nonoverlapping(
                            ecmascript_src.as_u8().unwrap(),
                            ecmascript_dst.as_u8().unwrap(),
                            len,
                        );
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
    const ARENA_SIZE_IN_WORDS: usize = (u8::MAX as usize) * 8;
    let mut rust_dst = Box::new([0usize; ARENA_SIZE_IN_WORDS]);
    // SAFETY: can transmute usize to u8s.
    let (head, rust_dst, tail) = unsafe { rust_dst.align_to_mut::<u8>() };
    assert!(head.is_empty() && rust_dst.len() == ARENA_SIZE && tail.is_empty());

    let mut ecmascript_dst = Box::new([0usize; ARENA_SIZE_IN_WORDS]);
    {
        // SAFETY: can transmute usize to u8s.
        let (head, body, tail) = unsafe { ecmascript_dst.align_to_mut::<u8>() };
        assert!(head.is_empty() && body.len() == ARENA_SIZE && tail.is_empty());
    }
    // SAFETY: Leaked box allocation is valid for reads and writes; length was
    // checked.
    let ecmascript_dst = unsafe {
        RacyAtomicSlice::enter(NonNull::from(Box::leak(ecmascript_dst)).cast(), ARENA_SIZE)
    };

    execute_ops(rust_dst, ecmascript_dst.slice(0, usize::MAX), &input.ops);

    // SAFETY: Racy atomics have finished (and actually no racing was performed
    // as all the ops tests are single-threaded). We can reallocate the memory
    // as Rust memory again.
    let (ecmascript_dst, _) = unsafe { ecmascript_dst.exit() };

    // SAFETY: The pointer is indeed pointing to a memory of this type.
    let _ = unsafe {
        Box::from_raw(
            ecmascript_dst
                .cast::<[usize; ARENA_SIZE_IN_WORDS]>()
                .as_ptr(),
        )
    };
});
