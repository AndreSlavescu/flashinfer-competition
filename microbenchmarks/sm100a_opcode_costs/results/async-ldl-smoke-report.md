# SM100A Opcode Cost Confirmation

- Total entries: 44
- Sync entries: 13
- Async entries: 31
- Direct evidence rows: 32
- Proxy rows: 8
- Unverified rows: 4
- Confirmed within +/-1 cycle(s): 1

## Mismatches

- None

## Direct

- `FFMA`: measured  vs expected 4 (observed `FFMA` via ``)
- `FADD`: measured  vs expected 4 (observed `FADD` via ``)
- `FMUL`: measured  vs expected 4 (observed `FMUL` via ``)
- `HFMA2`: measured  vs expected 4 (observed `HFMA2` via ``)
- `MUFU`: measured  vs expected 8 (observed `MUFU` via ``)
- `IMAD`: measured  vs expected 4 (observed `IMAD` via ``)
- `IADD3`: measured  vs expected 4 (observed `IADD3` via ``)
- `LOP3`: measured  vs expected 4 (observed `LOP3` via ``)
- `LDS`: measured  vs expected 25 (observed `LDS` via ``)
- `SHFL`: measured  vs expected 4 (observed `SHFL` via ``)
- `S2R`: measured  vs expected 20 (observed `S2R` via ``)
- `UTCHMMA`: measured  vs expected 1 (observed `UTCHMMA` via ``)
- `UTMALDG`: measured  vs expected 1 (observed `UTMALDG` via ``)
- `UTMASTG`: measured  vs expected 1 (observed `UTMASTG` via ``)
- `UTMAREDG`: measured  vs expected 1 (observed `UTMAREDG` via ``)
- `UTMAPF`: measured  vs expected 1 (observed `UTMAPF` via ``)
- `UTCCP`: measured  vs expected 1 (observed `UTCCP` via ``)
- `LDT`: measured  vs expected 1 (observed `LDT` via ``)
- `LDTM`: measured  vs expected 1 (observed `LDTM` via ``)
- `STT`: measured  vs expected 1 (observed `STT` via ``)
- `STTM`: measured  vs expected 1 (observed `STTM` via ``)
- `LDGSTS`: measured  vs expected 1 (observed `LDGSTS` via ``)
- `CCTL`: measured  vs expected 1 (observed `CCTL` via ``)
- `UBLKCP`: measured  vs expected 1 (observed `UBLKCP` via ``)
- `UBLKPF`: measured  vs expected 1 (observed `UBLKPF` via ``)
- `UBLKRED`: measured  vs expected 1 (observed `UBLKRED` via ``)
- `LDG`: measured  vs expected 1 (observed `LDG` via ``)
- `STG`: measured  vs expected 1 (observed `STG` via ``)
- `LDL`: measured 0 vs expected 1 (observed `LDL` via `async_ldl`)
- `STL`: measured  vs expected 1 (observed `STL` via ``)
- `MBAR`: measured  vs expected 1 (observed `SYNCS.ARRIVE` via ``)
- `BAR`: measured  vs expected 20 (observed `BAR` via ``)

## Proxy

- `STS`: measured  vs expected 25 (observed `STS` via ``; proxied via dependent STS+LDS chain)
- `UTCIMMA`: measured  vs expected 1 (observed `UTCHMMA` via `async_utchmma`; proxied with f16 tcgen05.mma issue kernel)
- `UTCOMMA`: measured  vs expected 1 (observed `UTCHMMA` via `async_utchmma`; proxied with f16 tcgen05.mma issue kernel)
- `UTCQMMA`: measured  vs expected 1 (observed `UTCHMMA` via `async_utchmma`; proxied with f16 tcgen05.mma issue kernel)
- `UTMACCTL`: measured  vs expected 1 (observed `UTMAPF` via `async_tma_prefetch`; proxied with TMA prefetch/cache-control path)
- `UTCSHIFT`: measured  vs expected 1 (observed `LDTM` via `async_ldtm`; proxied with TMEM rearrangement/load path)
- `LDGMC`: measured  vs expected 1 (observed `LDG` via `async_ldg`; proxied with global load/cache-control path)
- `CCTLL`: measured  vs expected 1 (observed `CCTL` via ``; ptxas selected short-form cache-control encoding)

## Unverified

- `MOV`: no qualifying measurement+evidence pair
- `LDGDEPBAR`: no qualifying measurement+evidence pair
- `STAS`: no qualifying measurement+evidence pair
- `REDAS`: no qualifying measurement+evidence pair
