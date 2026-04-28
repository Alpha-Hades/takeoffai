import streamlit as st
import anthropic
import base64
import json
import pandas as pd
from PIL import Image
import io
import fitz

st.set_page_config(page_title="TakeoffAI", page_icon="🏗️", layout="wide")

st.title("🏗️ TakeoffAI")
st.caption(
    "Upload an HVAC blueprint PDF. AI scans every page and builds your Bill of Materials."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "API Key", type="password", help="Get yours free at console.anthropic.com"
    )
    st.divider()
    safety_pct = st.selectbox(
        "Safety buffer",
        [0, 5, 10, 15],
        index=1,
        format_func=lambda x: f"+{x}% waste factor" if x else "None (0%)",
    )
    inc_low = st.toggle("Include low-confidence items", value=True)
    st.divider()
    st.caption("Powered by Claude AI")

uploaded = st.file_uploader(
    "Upload blueprint PDF",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Any HVAC, mechanical, or M-sheet drawing",
)

if not uploaded:
    st.info("Upload a blueprint PDF above to get started.")
    st.stop()

if not api_key:
    st.warning("Enter your API key in the sidebar.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)


@st.cache_data
def get_pages(file_bytes, file_type):
    images = []
    if file_type == "application/pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    else:
        images.append(Image.open(io.BytesIO(file_bytes)))
    return images


with st.spinner("Reading PDF pages..."):
    pages = get_pages(uploaded.read(), uploaded.type)

st.success(f"Loaded {len(pages)} page(s). Select a page and click Analyze.")

page_idx = st.selectbox(
    "Select page to analyze", range(len(pages)), format_func=lambda i: f"Page {i + 1}"
)

img = pages[page_idx]

col_img, col_bom = st.columns([1.2, 1])

with col_img:
    st.subheader(f"Blueprint — Page {page_idx + 1}")
    st.image(img, use_container_width=True)

with col_bom:
    st.subheader("Bill of Materials")

    if st.button("🔍 Analyze this page", type="primary", use_container_width=True):

        def img_to_b64(image):
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()

        prompt = """You are an expert HVAC estimator doing a quantity takeoff on a mechanical drawing.

Identify and count EVERY component visible on this drawing page.

Look for:
- Supply and return air diffusers and grilles — note size and shape
- Air handling units, fan coil units, VAV boxes
- Exhaust fans and ventilation fans
- Ductwork runs — estimate linear feet
- Thermostats, sensors, controls
- Any equipment schedule tables — extract them row by row
- Piping runs, valves, pumps if visible

Return ONLY a valid JSON array. No other text. No markdown. No explanation. Just the raw JSON array starting with [ and ending with ].

Each object must have exactly these fields:
{
  "item_code": "short code e.g. SQ-24 or AHU-1",
  "description": "full description",
  "quantity": <integer>,
  "unit": "EA or LF or SF",
  "confidence": "high or medium or low",
  "notes": "size, model number, or anything notable — empty string if nothing"
}

Only include items you can clearly see. Be conservative. If there is an equipment schedule table on the page, extract every single row from it."""

        with st.spinner("Claude is scanning the drawing... (~20 seconds)"):
            try:
                import time

                t0 = time.time()

                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=2000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": img_to_b64(img),
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )

                elapsed = round(time.time() - t0, 1)
                raw = response.content[0].text.strip()

                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                items = json.loads(raw.strip())
                st.session_state["items"] = items
                st.session_state["elapsed"] = elapsed
                st.session_state["page"] = page_idx
                st.success(f"Found {len(items)} items in {elapsed}s")

            except json.JSONDecodeError:
                st.error("Unexpected format returned. Try again.")
            except Exception as e:
                st.error(f"Error: {e}")

    if "items" in st.session_state and st.session_state.get("page") == page_idx:
        items = st.session_state["items"]
        elapsed = st.session_state.get("elapsed", "—")
        mult = 1 + safety_pct / 100

        visible = [i for i in items if inc_low or i["confidence"] != "low"]

        total_units = sum(round(i["quantity"] * mult) for i in visible)
        high_count = sum(1 for i in visible if i["confidence"] == "high")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Line items", len(visible))
        c2.metric("Total units", total_units)
        c3.metric("High confidence", f"{high_count}/{len(visible)}")
        c4.metric("Analyzed in", f"{elapsed}s")

        st.divider()

        rows = []
        for idx, item in enumerate(visible):
            qty = round(item["quantity"] * mult)
            rows.append(
                {
                    "#": idx + 1,
                    "Item code": item["item_code"],
                    "Description": item["description"],
                    "Qty": qty,
                    "Unit": item["unit"],
                    "Confidence": item["confidence"],
                    "Notes": item.get("notes", ""),
                }
            )

        df = pd.DataFrame(rows)

        edited = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Item code": st.column_config.TextColumn(width="small"),
                "Description": st.column_config.TextColumn(width="large"),
                "Qty": st.column_config.NumberColumn(width="small", min_value=0),
                "Unit": st.column_config.TextColumn(width="small"),
                "Confidence": st.column_config.TextColumn(width="small"),
                "Notes": st.column_config.TextColumn(width="medium"),
            },
            num_rows="dynamic",
        )

        if safety_pct:
            st.caption(f"Quantities include +{safety_pct}% safety buffer")

        st.divider()

        csv = edited.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV — Bill of Materials",
            data=csv,
            file_name=f"takeoff_page{page_idx + 1}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )