https://ai.google.dev/gemini-api/docs/caching?lang=python

# Gemini Cache Break-even Formula

## 1. Parameters
- **S** = Size of system prompt (tokens)  
- **N** = Number of reuses within TTL (requests)  
- **Cin** = Normal input cost per 1M tokens  
- **Cimp** = Implicit cache cost per 1M tokens (~25% of Cin)  
- **Cexp** = Explicit cache cost per 1M tokens  
- **Cstor** = Explicit cache storage cost per 1M tokens per hour  
- **T** = TTL (hours)  

---

## 2. Cost Formulas
### No Cache
$$
Cost_{no} = N \times \frac{S}{10^6} \times C_{in}
$$

### Implicit Cache
$$
Cost_{imp} = N \times \frac{S}{10^6} \times C_{imp}
$$

### Explicit Cache
$$
Cost_{exp} =
\underbrace{\frac{S}{10^6} \times C_{in}}_{\text{create once}}
+ N \times \frac{S}{10^6} \times C_{exp}
+ \frac{S}{10^6} \times C_{stor} \times T
$$

---

## 3. Break-even Condition (Explicit vs No Cache)

Formula:
- N > (Cin + Cstor * T) / (Cin - Cexp)

Where:
- If **N** is larger than RHS (Right-Hand Side) → Explicit cache is cheaper  
- If smaller → No cache (or implicit) is better  

---

## 4. Example: Gemini 2.5 Pro (≤200k tokens)

- Cin = 1.25 $/M  
- Cexp = 0.31 $/M  
- Cstor = 4.50 $/M/hr  
- S = 20,000 tokens (~0.02M)  
- T = 2 hours  

Break-even:
- N > (1.25 + 4.5 * 2) / (1.25 - 0.31)
- N > (1.25 + 9) / 0.94
- N > 10.25 / 0.94
- N > ~11

👉 Need at least **11 reuses in 2 hours** for explicit cache to be worth it.

---

## 5. Rules of Thumb
- **Implicit caching first**: free, automatic, ~75% discount if prefix matches  
- **Explicit caching**: only if system prompt is long **and** reused enough within TTL  
- Larger **S** and higher **reuse frequency (N)** → more likely explicit is worth it  
- Short TTL + heavy traffic = best case for explicit  
- Few reuses or highly variable prompts → implicit is safer  

---
