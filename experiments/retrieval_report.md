# Retrieval / Blocking Methodology

## 1. Objective

The retrieval stage reduces the number of source-pair comparisons before pairwise entity matching.

For each Source-1 entity, the system retrieves a candidate set of potentially matching entities from Source-2 and Source-3.

The retrieval stage is designed for a many-to-many entity-resolution setting. Therefore, multiple target entities may be retained for one Source-1 entity.

A candidate retrieved by blocking is not considered a confirmed match. Final matching is performed by the downstream pairwise matching model.

---

## 2. Retrieval Architecture

The final retrieval system uses three complementary blocking views:

1. Normalized exact-name blocking
2. Normalized exact-address blocking
3. Informative name-token blocking

The candidate set is the UNION of all three views.

```text
                    Source-1
                       |
          +------------+------------+
          |            |            |
          v            v            v
        Name         Address       Name-token
       Blocking     Blocking       Blocking
          |            |            |
          +------------+------------+
                       |
                     UNION
                       |
                       v
              Candidate entities
                       |
                       v
               Pairwise matcher