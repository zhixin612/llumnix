| Decode (per layer)   | Compute Load  | Memory Access | Communication    |
| -------------------- | ------------- | ------------- | ---------------- |
| 1. Embedding(Once)   | $2BvH$        | $vH$          | $0$              |
| 2. LayerNorm         | $4BSH(RMS)$   | $4BSH$        | $0$              |
| 3. QKV               | $2BHH+4BHnd$  | $HH+2Hnd$     | $0$              |
| 4. Attention         | $4BSH$        | $0$           | $0$              |
| 5. (mem)KVcahce      | $0$           | $2Bnd$        | $0$              |
| 6. (comm) AllGather  | $0$           | $0$           | $4BH/tp$         |
| 7. O_proj            | $2BHH$        | $HH$          | $0$              |
| 8. (comm) AllReduce  | $0$           | $0$           | $BH(tp-1)/tp$    |
| 9. LayerNorm         | $4BSH(RMS)$   | $4BH$         | $0$              |
| 10. Up and Gate      | $4BHI / 2BHI$ | $HI / 2HI$    | $0$              |
| 11. Gate*Up          | $BI$          | $0$           | $0$              |
| 12. (mem)activation  | $0$           | $2BI$         | $0$              |
| 13. Down             | $2BHI$        | $HI$          | $0$              |
| 14. (comm) AllGather | $0$           | $0$           | $2BH(tp-1)/tp^2$ |
| 15. Embedding(Once)  | $2BvH$        | $vH$          | $0$              |





| Prefill (per layer)  | Compute Load    | Memory Access | Communication     |
| -------------------- | --------------- | ------------- | ----------------- |
| 1. Embedding(Once)   | $2BSvH$         | $SvH$         | $0$               |
| 2. LayerNorm         | $4BSH(RMS)$     | $4BSH$        | $0$               |
| 3. QKV               | $2BSHH+4BHSnd$  | $HH+2Hnd$     | $0$               |
| 4. Attention         | $4BSSH$         | $0$           | $0$               |
| 5. (mem)KVcahce      | $0$             | $2BSnd$       | $0$               |
| 6. (comm) AllGather  | $0$             | $0$           | $4BSH/tp$         |
| 7. O_proj            | $2BSHH$         | $HH$          | $0$               |
| 8. (comm) AllReduce  | $0$             | $0$           | $2BSH$            |
| 9. LayerNorm         | $4BSH(RMS)$     | $4BSH$        | $0$               |
| 10. Up and Gate      | $4BSHI / 2BSHI$ | $HI / 2HI$    | $0$               |
| 11. Gate*Up          | $BSI$           | $0$           | $0$               |
| 12. (mem)activation  | $0$             | $2BSI$        | $0$               |
| 13. Down             | $2BSHI$         | $HI$          | $0$               |
| 14. (comm) AllGather | $0$             | $0$           | $2BSH$            |  
| 15. Embedding(Once)  | $2BSvH$         | $SvH$         | $0$               |
