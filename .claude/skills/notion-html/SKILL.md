---
name: notion-html
description: Publish a Notion page carrying interactive HTML, live JS widgets, LaTeX equations, or images — the upload route that actually works, the embed syntax, what cannot be verified, and the page hygiene rules. Use whenever writing or updating a Notion page for this project, especially one with diagrams, widgets, figures or maths.
---

# Publishing HTML, JS and LaTeX to Notion

Notion 3.6 (2026-07-01) added **HTML blocks**: an HTML attachment rendered inline in a
sandboxed iframe, with **live JavaScript**. That capability is newer than most training
data, so the default assumption "Notion cannot run JS" is wrong — it can, and this project
uses it for interactive explainers.

Everything below is measured on this machine, not inferred.

## 1. The upload route — `create-attachment`, never curl

**Use `mcp__claude_ai_Notion__notion-create-attachment` with inline `content`.** It goes
through Notion's own infrastructure and returns a `file-upload://` source.

**Do not upload HTML by curl.** `create_file_upload`'s direct endpoint sits behind
Cloudflare, and **any HTML body containing `<script>` returns 403**. This was isolated with
a 118-byte trivial file, so the trigger is the *script tag*, not size or complexity. SVGs
and binaries pass fine.

Two further wrinkles:

- The **sandbox proxy itself trips Cloudflare**, so even non-script uploads by curl need
  `dangerouslyDisableSandbox`. Another reason to use the MCP tool instead.
- `gh pr edit` failing with a Projects-classic GraphQL error is a *different* problem on
  the kinder repos — unrelated, don't conflate them.

## 2. The embed syntax

```
<embed src="{{file-upload://...}}" color?="Color">Caption</embed>
```

"HTML", "HTML block", "HTML artifact" and "HTML embed" all mean *an HTML attachment
rendered with `<embed>`*. **Never create one as a code block or a file block** — those
render as inert text or a download link, not a running page.

The authoritative spec is the MCP resource `notion://docs/enhanced-markdown-spec`. Read it
via `notion-fetch` with that URI as the `id` if you need the current details; it changes.

## 3. Test one small embed first

Build a trivial widget, upload it, **re-fetch the page, and confirm it comes back as an
`<embed>` block holding the attachment** — before building four more. A silent downgrade to
an inert block is the failure you are checking for, and finding it on widget five is
expensive.

### `content_length` is not proof of byte-exactness

The tool takes only **inline `content`**, so there is no mechanical path from a file on disk
to the uploaded bytes — a model has to emit them. That introduces one specific corruption
you cannot avoid and must not mis-verify:

**Decomposed Unicode gets NFC-normalized in transit.** Measured: `s` + U+0307 (bytes
`73 cc 87`) comes back as precomposed U+1E61 (`e1 b9 a1`). `q̇`, `d̂` and `D̂` survive
decomposed only because no precomposed codepoint exists for them; `ṡ` never does.

Both sequences are **3 bytes**, so `NFC(original)` has **exactly the same length** as the
original while differing at 3 byte offsets. **A `content_length` match therefore passes on a
file that is not byte-exact.** Compare **sha256** instead, and expect the NFC form.

The practical fix is to **NFC-normalize the source file itself** before uploading, then
re-run tests and screenshots and re-serve, so the disk file, the served copy and the Notion
attachment all share one sha. That makes the artifact reproducible through any text pipeline
rather than carrying a character no model can round-trip. Do this to the source rather than
recording two different shas.

Note this is cosmetic in rendering terms — the affected character displays identically — so
the cost of getting it wrong is a false provenance claim, not a broken page. Say "verified
identical after NFC normalization" rather than "byte-exact" when that is what happened.

## 4. What you cannot verify, and what to do instead

**You have no browser into Notion**, so you cannot confirm the sandboxed iframe actually
*executes* the JS. Do not claim it does. Instead:

1. Test the widget properly outside Notion — **jsdom** for the controls (every button,
   slider and checkbox fires; no exceptions thrown; readouts actually change) and
   **headless Firefox** for the render. This catches real defects: one pass found
   overlapping axis labels that no amount of code review would have.
2. Confirm the `<embed>` blocks round-trip by re-fetching.
3. Serve **byte-equivalent copies** on the scratch web server and hand Josh full clickable
   `http://agni:8765/<file>` URLs, one per artifact — those demonstrably run.
4. Say plainly in your report which parts are verified and which are not.

## 5. LaTeX

Notion renders maths natively: `$...$` inline, `$$...$$` display. Prefer it to an image of
an equation — it reflows, it is searchable, and it survives theme changes.

**Colour-coded maths** (`\textcolor{#RRGGBB}{...}` / `\color{...}`) is KaTeX syntax that
Notion *may* support — **treat it as unconfirmed and test one equation before committing to
a colour-coded page.** If it does not survive, fall back to coloured inline HTML spans
around symbol names in prose, plus a legend, and say which you used.

## 6. Figures

Static figures upload the same way. Prefer SVG where the content is line art (it scales and
stays crisp); PNG for anything rasterised. Both pass Cloudflare by curl, but
`create-attachment` is still the simpler path.

Check any palette against **both light and dark Notion backgrounds** — readers' themes
differ and a page that only works in one is half-broken.

**Do not reuse this project's blue/orange figure palette for explanatory diagrams.** In
this repo `#0072B2` / `#D55E00` encode *reset-vs-never arms*; using them for anything else
invites a reader to see an arm comparison that isn't there. Pick a semantic palette for
explanatory work and **say on the page that you deviated and why**.

## 7. Page hygiene — what not to touch

- **Never edit another agent's page.** Several agents write Notion concurrently here.
- **Never touch a parent page's `# Claude Plan` section** or its inline comment threads.
  Verify by **re-fetching and diffing**, not by trusting a tool's return status —
  `notion-fetch` with `include_discussions: true` reports a `discussion-count` you can
  check against what was there before.
- Append to an existing page rather than rewriting it, unless asked to rewrite.
- Prefer **cross-linking** to duplicating: if a sibling page already explains something,
  build on it.

## 8. Status semantics on the Global Tasks Tracker

Allowed values, fetched from the data source rather than guessed: `Not started`, `Blocked`,
`In Review`, `In progress`, `Done`.

**`Blocked` means blocked on someone *external*** — a reviewer, an upstream maintainer, a
third party. **A task waiting on Josh himself stays `In progress`.** Sequencing behind
another of our own tasks is an internal decision, so also `In progress`. The board is
Josh's queue; marking his own pending decisions `Blocked` hides them among the genuinely
unactionable ones and makes the board read as more stuck than it is.

## 9. Report the page name

Josh asks for the page title every time. Give the **title and the URL** in your report, not
just the URL.
