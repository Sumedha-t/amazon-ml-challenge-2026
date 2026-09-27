from src.matching.local_graph import build_local_cross_source_graph


s2_candidates = [
    {
        "entity_id": "S2-A",
        "business_name": "Team Air Pvt Ltd",
        "business_address": "Flat No 101, Anuska Towers, Bangalore",
        "country": "India",
    },
    {
        "entity_id": "S2-B",
        "business_name": "Completely Different Company",
        "business_address": "Mumbai Industrial Area",
        "country": "India",
    },
]


s3_candidates = [
    {
        "entity_id": "S3-A",
        "business_name": "Team Air Private Limited",
        "business_address": "101 Anuska Towers, Bangalore",
        "country": "India",
    },
    {
        "entity_id": "S3-B",
        "business_name": "Another Different Company",
        "business_address": "Mumbai Industrial Area",
        "country": "India",
    },
]


edges = build_local_cross_source_graph(
    s2_candidates,
    s3_candidates,
)


print(f"Number of S2-S3 edges: {len(edges)}")

for edge in edges:
    print(
        f"{edge.s2_id} <-> {edge.s3_id} | "
        f"support={edge.support:.3f} | "
        f"contradiction={edge.contradiction:.1f}"
    )