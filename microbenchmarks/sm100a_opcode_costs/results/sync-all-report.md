# SM100A Opcode Cost Confirmation

- Total entries: 44
- Sync entries: 13
- Async entries: 31
- Direct evidence rows: 32
- Proxy rows: 8
- Unverified rows: 4
- Confirmed within +/-1 cycle(s): 12

## Mismatches

- `MOV`: expected 4, measured 0, delta -4 (unverified)

## Direct

- `FFMA`: measured 4 vs expected 4 (observed `FFMA` via `sync_ffma`)
- `FADD`: measured 4 vs expected 4 (observed `FADD` via `sync_fadd`)
- `FMUL`: measured 4 vs expected 4 (observed `FMUL` via `sync_fmul`)
- `HFMA2`: measured 4 vs expected 4 (observed `HFMA2` via `sync_hfma2`)
- `MUFU`: measured 17 vs expected 17 (observed `MUFU` via `sync_mufu`)
- `IMAD`: measured 4 vs expected 4 (observed `IMAD` via `sync_imad`)
- `IADD3`: measured 2 vs expected 2 (observed `IADD3` via `sync_iadd3`)
- `LOP3`: measured 4 vs expected 4 (observed `LOP3` via `sync_lop3`)
- `LDS`: measured 41 vs expected 41 (observed `LDS` via `sync_lds`)
- `SHFL`: measured 26 vs expected 26 (observed `SHFL` via `sync_shfl`)
- `S2R`: measured 19 vs expected 19 (observed `S2R` via `sync_s2r`)
- `UTCHMMA`: measured  vs expected 54 (observed `UTCHMMA` via ``)
- `UTMALDG`: measured  vs expected 110 (observed `UTMALDG` via ``)
- `UTMASTG`: measured  vs expected 14 (observed `UTMASTG` via ``)
- `UTMAREDG`: measured  vs expected 45 (observed `UTMAREDG` via ``)
- `UTMAPF`: measured  vs expected 57 (observed `UTMAPF` via ``)
- `UTCCP`: measured  vs expected 70 (observed `UTCCP` via ``)
- `LDT`: measured  vs expected 0 (observed `LDT` via ``)
- `LDTM`: measured  vs expected 0 (observed `LDTM` via ``)
- `STT`: measured  vs expected 2 (observed `STT` via ``)
- `STTM`: measured  vs expected 54 (observed `STTM` via ``)
- `LDGSTS`: measured  vs expected 6 (observed `LDGSTS` via ``)
- `CCTL`: measured  vs expected 4 (observed `CCTL` via ``)
- `UBLKCP`: measured  vs expected 15 (observed `UBLKCP` via ``)
- `UBLKPF`: measured  vs expected 6 (observed `UBLKPF` via ``)
- `UBLKRED`: measured  vs expected 44 (observed `UBLKRED` via ``)
- `LDG`: measured  vs expected 0 (observed `LDG` via ``)
- `STG`: measured  vs expected 21 (observed `STG` via ``)
- `LDL`: measured  vs expected 0 (observed `LDL` via ``)
- `STL`: measured  vs expected 4 (observed `STL` via ``)
- `MBAR`: measured  vs expected 7 (observed `SYNCS.ARRIVE` via ``)
- `BAR`: measured  vs expected 20 (observed `BAR` via ``)

## Proxy

- `STS`: measured 10 vs expected 10 (observed `STS` via `sync_sts`; {"target": 60.001, "baseline": 50.001, "proxy": false} | proxied via dependent STS+LDS chain)
- `UTCIMMA`: measured  vs expected 54 (observed `UTCHMMA` via `async_utchmma`; proxied with f16 tcgen05.mma issue kernel)
- `UTCOMMA`: measured  vs expected 54 (observed `UTCHMMA` via `async_utchmma`; proxied with f16 tcgen05.mma issue kernel)
- `UTCQMMA`: measured  vs expected 54 (observed `UTCHMMA` via `async_utchmma`; proxied with f16 tcgen05.mma issue kernel)
- `UTMACCTL`: measured  vs expected 57 (observed `UTMAPF` via `async_tma_prefetch`; proxied with TMA prefetch/cache-control path)
- `UTCSHIFT`: measured  vs expected 0 (observed `LDTM` via `async_ldtm`; proxied with TMEM rearrangement/load path)
- `LDGMC`: measured  vs expected 0 (observed `LDG` via `async_ldg`; proxied with global load/cache-control path)
- `CCTLL`: measured  vs expected 4 (observed `CCTL` via ``; ptxas selected short-form cache-control encoding)

## Unverified

- `MOV`: no qualifying measurement+evidence pair
- `LDGDEPBAR`: no qualifying measurement+evidence pair
- `STAS`: no qualifying measurement+evidence pair
- `REDAS`: no qualifying measurement+evidence pair
