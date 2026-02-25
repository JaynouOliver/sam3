"""
Fetch product image URLs from Supabase for SAM3 auto-labeling.
READ-ONLY — no writes to the database.
"""
import psycopg2
import json

# Direct connection
conn = psycopg2.connect(
    host="db.glfevldtqujajsalahxd.supabase.co",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="ML7tduJJc6wgaEt6",
)
conn.set_session(readonly=True)

cursor = conn.cursor()

query = """
SELECT
  p."id",
  p."name",
  p."product_group_id",
  p."productType",
  p.metadata->'materialData'->'files'->>'color_original' AS upscaled_image,
  p."materialData"->'renderedImage' AS rendered_image
FROM public."productsV2" AS p
WHERE
  p."productType" IN ('fixed material', 'material', 'paint', 'hardware', 'accessory')
  AND p."objectStatus" IN ('APPROVED', 'APPROVED_PRO')
"""

print("Running query (read-only)...")
cursor.execute(query)
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]

print(f"Found {len(rows)} products\n")

# Show first 5 rows
for i, row in enumerate(rows[:5]):
    print(f"--- Row {i+1} ---")
    for col, val in zip(columns, row):
        print(f"  {col}: {val}")
    print()

# Summary by productType
cursor.execute("""
SELECT p."productType", COUNT(*)
FROM public."productsV2" AS p
WHERE
  p."productType" IN ('fixed material', 'material', 'paint', 'hardware', 'accessory')
  AND p."objectStatus" IN ('APPROVED', 'APPROVED_PRO')
GROUP BY p."productType"
ORDER BY COUNT(*) DESC
""")
print("Breakdown by productType:")
for row in cursor.fetchall():
    print(f"  {row[0]:20s} {row[1]:>6}")

# Count how many have image URLs
has_upscaled = sum(1 for r in rows if r[4])
has_rendered = sum(1 for r in rows if r[5])
print(f"\nWith upscaled_image URL: {has_upscaled}/{len(rows)}")
print(f"With rendered_image:     {has_rendered}/{len(rows)}")

# Save results to JSON for downstream use
results = []
for row in rows:
    results.append({
        "id": str(row[0]),
        "name": row[1],
        "product_group_id": str(row[2]) if row[2] else None,
        "productType": row[3],
        "upscaled_image": row[4],
        "rendered_image": row[5] if row[5] else None,
    })

output_path = "/teamspace/studios/this_studio/products_for_labeling.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {output_path}")

cursor.close()
conn.close()
print("Connection closed.")
