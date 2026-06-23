import { useState, useRef, useCallback } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [indexed, setIndexed] = useState(0);
  const [searching, setSearching] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [searched, setSearched] = useState(false);
  const fileRef = useRef(null);

  const uploadFiles = useCallback(async (files) => {
    const fileList = Array.from(files || []);
    if (!fileList.length) return;

    setUploading(true);

    for (const file of fileList) {
      if (!file.type.startsWith("image/")) continue;

      const form = new FormData();
      form.append("file", file);

      try {
        await axios.post(`${API}/upload`, form);
        setIndexed((n) => n + 1);
      } catch (e) {
        console.error("Upload failed for", file.name, e);
      }
    }

    setUploading(false);
  }, []);

  const runSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) return;

    setSearching(true);
    setSearched(true);

    try {
      const { data } = await axios.get(`${API}/search`, {
        params: { q, k: 6 },
      });

      const items = Array.isArray(data) ? data : data.results || [];
      setResults(items);
    } catch (e) {
      console.error("Search failed", e);
      setResults([]);
    }

    setSearching(false);
  }, [query]);

  const fileUrl = (path) => {
    const clean = String(path)
      .replaceAll("\\", "/")
      .replace(/^\.?\//, "");

    return `${API}/${clean}`;
  };

  return (
    <div className="app">
      <style>{css}</style>

      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" aria-hidden />
          <span className="brand-name">latent</span>
        </div>
        <div className="counter">
          <span className="counter-num">{indexed}</span>
          <span className="counter-label">images added this session</span>
        </div>
      </header>

      <main className="stage">
        <section className="hero">
          <h1 className="title">Photo Search</h1>
          <p className="lede">
            Upload images and search them with natural language. The closest images show up, ranked by how well they match.
            <br />
            <br />
            Some images have already been uploaded. You can try: &ldquo;a boat on water&rdquo;, or &ldquo;person playing instrument&rdquo;, etc.
          </p>

          <div className="searchbar">
            <input
              className="search-input"
              placeholder="Describe the photo you're looking for"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runSearch()}
            />
            <button
              className="search-go"
              onClick={runSearch}
              disabled={searching || !query.trim()}
            >
              {searching ? "Searching" : "Search"}
            </button>
          </div>

          <div
            className={`adder ${dragging ? "adder--on" : ""}`}
            onDragEnter={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDragging(true);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDragging(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDragging(false);
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDragging(false);
              uploadFiles(e.dataTransfer.files);
            }}
            onClick={() => fileRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(e) => uploadFiles(e.target.files)}
            />
            {uploading
              ? "Indexing images..."
              : "Drop images here, or click to add them to the index"}
          </div>
        </section>

        <section className="results">
          {searched && !searching && results.length === 0 && (
            <p className="empty">
              No matches yet. Add some images to the index, then search.
            </p>
          )}

          <div className="grid">
            {results.map((r, i) => {
              const pct = Math.round((r.score ?? 0) * 100);
              return (
                <figure className="card" key={`${r.path}-${i}`}>
                  <div className="thumb">
                    <img src={fileUrl(r.path)} alt="" loading="lazy" />
                  </div>
                  <figcaption className="meta">
                    <div className="bar">
                      <div
                        className="bar-fill"
                        style={{ width: `${Math.max(pct, 4)}%` }}
                      />
                    </div>
                    <span className="pct">{pct}%</span>
                  </figcaption>
                </figure>
              );
            })}
          </div>
        </section>
      </main>
    </div>
  );
}

const css = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

.app {
  min-height: 100vh;
  background:
    radial-gradient(1100px 600px at 50% -15%, rgba(143, 200, 255, 0.10), transparent 60%),
    #0a1628;
  color: #eaf1fb;
  font-family: 'Inter', system-ui, sans-serif;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 22px 56px;
  border-bottom: 1px solid #1e3457;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: #8fc8ff;
  box-shadow: 0 0 14px rgba(143, 200, 255, 0.6);
}

.brand-name {
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.counter {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.counter-num {
  color: #8fc8ff;
  font-weight: 700;
}

.counter-label {
  color: #647ca0;
  font-size: 12px;
  letter-spacing: 0.04em;
}

.stage {
  max-width: 1080px;
  margin: 0 auto;
  padding: 0 56px;
}

.hero {
  padding: 80px 0 48px;
  max-width: 720px;
}

.title {
  font-size: 48px;
  font-weight: 300;
  line-height: 1.08;
  letter-spacing: -0.025em;
  margin: 0 0 20px;
}

.lede {
  color: #9fb3d1;
  line-height: 1.6;
  margin-bottom: 36px;
}

.searchbar {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #0f1f38;
  border: 1px solid #1e3457;
  border-radius: 14px;
  padding: 8px;
}

.searchbar:focus-within {
  border-color: #3a86d4;
  box-shadow: 0 0 0 4px rgba(58, 134, 212, 0.18);
}

.search-input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: #eaf1fb;
  font-family: inherit;
  font-size: 16px;
  padding: 12px;
}

.search-input::placeholder {
  color: #647ca0;
}

.search-go {
  background: #8fc8ff;
  color: #0a1628;
  border: none;
  border-radius: 9px;
  padding: 13px 26px;
  font-family: inherit;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.16s, transform 0.1s;
}

.search-go:hover:not(:disabled) {
  background: #bfe0ff;
}

.search-go:active:not(:disabled) {
  transform: translateY(1px);
}

.search-go:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.adder {
  margin-top: 14px;
  border: 1.5px dashed #1e3457;
  border-radius: 12px;
  padding: 18px;
  text-align: center;
  cursor: pointer;
  color: #647ca0;
  font-size: 14px;
  transition: border-color 0.18s, color 0.18s, background 0.18s;
}

.adder:hover {
  border-color: #3a86d4;
  color: #9fb3d1;
}

.adder--on {
  border-color: #8fc8ff;
  background: rgba(143, 200, 255, 0.06);
  color: #8fc8ff;
}

.results {
  padding-bottom: 72px;
  min-height: 120px;
}

.empty {
  color: #647ca0;
  text-align: center;
  padding: 48px 0;
}

.grid {
  display: grid;
  gap: 18px;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
}

.card {
  margin: 0;
  background: #0f1f38;
  border: 1px solid #1e3457;
  border-radius: 14px;
  overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
}

.card:hover {
  border-color: #3a86d4;
  transform: translateY(-3px);
}

.thumb {
  aspect-ratio: 1;
  background: #16294a;
}

.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.meta {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 13px 14px 15px;
}

.bar {
  flex: 1;
  height: 5px;
  background: #16294a;
  border-radius: 99px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, #3a86d4, #8fc8ff);
}

.pct {
  font-size: 13px;
  font-weight: 600;
  color: #8fc8ff;
  min-width: 38px;
  text-align: right;
}

@media (max-width: 540px) {
  .topbar {
    padding: 18px 24px;
  }

  .stage {
    padding: 0 24px;
  }

  .hero {
    padding-top: 54px;
  }

  .title {
    font-size: 36px;
  }

  .searchbar {
    flex-wrap: wrap;
  }

  .search-go {
    width: 100%;
  }
}
`;