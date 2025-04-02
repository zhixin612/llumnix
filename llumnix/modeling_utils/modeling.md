| Decode (per layer)   | Compute Load  | Memory Access | Communication    |
| -------------------- | ------------- | ------------- | ---------------- |
| 1. Embedding(Once)   | $2BSvH$       | $SvH$         | $0$              |
| 2. LayerNorm         | $4BSH(RMS)$   | $4BSH$        | $0$              |
| 3. QKV               | $2BHH+4BHnd$  | $HH+2Hnd$     | $0$              |
| 4. Attention         | $4BSH$        | $0$           | $0$              |
| 5. (comm) AllGather  | $0$           | $0$           | $BH(tp-1)/tp$    |
| 6. O_proj            | $2BHH$        | $HH$          | $0$              |
| 7. (comm) AllGather  | $0$           | $0$           | $BH(tp-1)/tp$    |
| 8. LayerNorm         | $4BSH(RMS)$   | $4BSH$        | $0$              |
| 9. Up and Gate       | $4BHI / 2BHI$ | $HI / 2HI$    | $0$              |
| 10. Gate*Up          | $BI$          | $0$           | $0$              |
| 11. Down             | $2BHI$        | $HI$          | $0$              |
| 12. (comm) AllReduce | $0$           | $0$           | $2BH(tp-1)/tp^2$ |
| 13. LayerNorm        | $4BSH(RMS)$   | $4BSH$        | $0$              |
| 14. Embedding(Once)  | $2BSvH$       | $SvH$         | $0$              |





| Prefill (per layer)  | Compute Load    | Memory Access | Communication     |
| -------------------- | --------------- | ------------- | ----------------- |
| 1. Embedding(Once)   | $2BSvH$         | $SvH$         | $0$               |
| 2. LayerNorm         | $4BSH(RMS)$     | $4BSH$        | $0$               |
| 3. QKV               | $2BSHH+4BHSnd$  | $HH+2Hnd$     | $0$               |
| 4. Attention         | $4BSSH$         | $0$           | $0$               |
| 5. (comm) AllGather  | $0$             | $0$           | $BSH(tp-1)/tp$    |
| 6. O_proj            | $2BSHH$         | $HH$          | $0$               |
| 7. (comm) AllGather  | $0$             | $0$           | $BSH(tp-1)/tp$    |
| 8. LayerNorm         | $4BSH(RMS)$     | $4BSH$        | $0$               |
| 9. Up and Gate       | $4BSHI / 2BSHI$ | $HI / 2HI$    | $0$               |
| 10. Gate*Up          | $BSI$           | $0$           | $0$               |
| 11. Down             | $2BSHI$         | $HI$          | $0$               |
| 12. (comm) AllReduce | $0$             | $0$           | $2BSH(tp-1)/tp^2$ |
| 13. LayerNorm        | $4BSH(RMS)$     | $4BSH$        | $0$               |
| 14. Embedding(Once)  | $2BSvH$         | $SvH$         | $0$               |
