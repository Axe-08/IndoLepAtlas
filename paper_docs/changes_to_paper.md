# Changes to Paper — `indolep.tex`

Audit performed: 2026-04-30

---

## Spelling & Typographical Errors

| Line | Current | Correction |
|------|---------|------------|
| 69 | `assests` | `assets` |
| 69 | `INaturalist` | `iNaturalist` |
| 69 | `reknowned` | `renowned` |
| 69 | `Intercation` (MLFI) | `Interaction` |
| 77 | `photohraphs` | `photographs` |
| 85 | `Lepidotera` | `Lepidoptera` |
| 199 | `Leptosia Nina` | `Leptosia nina` (species epithet must be lowercase) |
| 256 | `catalouge` | `catalogue` |
| 630 | `Kalilima` (×2) | `Kallima` |
| 630 | `Inachus`, `Horsfieldii`, `Francisca`, `Gotama` | `inachus`, `horsfieldii`, `francisca`, `gotama` (species epithets lowercase) |
| 630 | `disambiguiation` | `disambiguation` |

---

## Grammar, Punctuation & Phrasing

| Line | Issue | Suggested Fix |
|------|-------|---------------|
| 77 | `This task though, fundamentally differs` — missing paired comma | `This task, though, fundamentally differs` |
| 85 | `images-all centered` — hyphen instead of em-dash | `images---all centered` |
| 203 | `comprises of` — incorrect usage | `comprises` or `consists of` |
| 252 | Straight quotes `"unknown"` | LaTeX curly quotes: ` ``unknown'' ` |
| 435 | Straight quotes `"furry surface textures"` etc. | LaTeX curly quotes: ` ``furry surface textures'' ` |
| 256 | `spot--checked`, `expert--curated` — double hyphen (en-dash) used for a compound adjective | Single hyphen: `spot-checked`, `expert-curated` |
| 365 | `than loss-function.` — missing article | `than the loss function.` |
| 400 | `demonstrating multi-level fusion does benefit` — missing "that" | `demonstrating that multi-level fusion does benefit` |
| 614 | `does not just adds parameters` — subject-verb disagreement | `does not just add parameters` |
| 630 | Comma splice: `...flight seasons, the additional context...` | `...flight seasons—the additional context the geotemporal embeddings provide for disambiguation.` |
| 635 | `region--based` — en-dash used for a compound adjective | Single hyphen: `region-based` |

---

## Missing References & Citations

| Line | Issue |
|------|-------|
| 94 | `Amarathunga et al. (2021)` — mentioned in text but **missing from bibliography** and not wrapped in `\cite{}` |
| 94 | `(Amarathunga et al. 2022)` — present in bibliography as `amarathunga2022thrips` but **not using `\cite{amarathunga2022thrips}`** in text |
| 94 | `Alfatemi et al. (2024)` — mentioned in text but **completely missing from bibliography** and not wrapped in `\cite{}` |
| 97 | `(Amarathunga et al. 2021)` — same as above; missing from bibliography, no `\cite{}` |
| 256 | `(cite book or catalouge)` — **placeholder text left in manuscript**, needs an actual `\cite{}` reference |
| 270 | `IP102` dataset in comparison table — **no citation** in the bibliography |

---

## Other Potential Issues

| Line | Issue |
|------|-------|
| 142 | Dataset link sentence has no period/full stop at the end before `\begin{figure}` |
| 274 | Two `\caption{}` commands inside a single `table` environment (lines 261 & 275) — this will produce two separate caption numbers under one table float. Consider splitting into two `\begin{table}` environments or using a single caption. |
| 335 | `\begin{table}` without `[H]` placement — may float away from the text. Other tables use `[H]`. |
| 371 | `\begin{table}` without `[H]` — same issue. |
| 406 | `\begin{table}` without `[H]` — same issue. |
| 410–411 | Table uses `\hline` while rest of paper uses `\toprule`/`\midrule`/`\bottomrule` — inconsistent style. |
| 43 | ORCID for Akshit Bansal is a placeholder `0000-1111-2222-3333` |

---

## LNCS Template Compliance Issues

Compared against `template/samplepaper.tex` (LLNCS Version 2.21, 2022/01/12).

### Critical — Will Likely Cause Rejection

| Line | Issue | Detail |
|------|-------|--------|
| 4 | **`\usepackage[margin=1in]{geometry}`** | The `llncs` class defines its own margins. Loading `geometry` **overrides the LNCS page layout** — this will be rejected. **Remove this line.** |
| 17 | **`\usepackage{fancyhdr}` + lines 32–35** | LNCS controls its own headers/footers (running author + running title). The `fancyhdr` override replaces them with just a page number in the right footer. **Remove `fancyhdr` and lines 32–35.** |
| 15 | **`\usepackage[numbers]{natbib}`** | LNCS provides its own bib style `splncs04.bst`. Using `natbib` can conflict with the class's citation formatting. **Use `splncs04.bst` instead**, or at minimum verify `natbib` doesn't break numbered citation format. |
| 687 | **`\bibliographystyle{plainnat}`** | Should be `splncs04` (the `.bst` file is in the template directory). `plainnat` produces author-year formatted entries, not the LNCS numeric style. |
| 691–725 | **`\bibitem` format uses `[Author, Year]` optional argument** | With LNCS `thebibliography`, these optional args produce author-year labels. The template uses plain `\bibitem{key}` with numeric labels. Either use `splncs04.bst` with BibTeX or remove the `[...]` optional arguments. |
| — | **Missing `\begin{credits}` section** | The template (lines 128–143) requires a `\begin{credits}` block before the bibliography containing `\ackname` (acknowledgements) and `\discintname` (competing interests disclosure). This is **mandatory** in current LNCS submissions. |
| — | **Missing `\authorrunning{}`** | The template (line 34) requires `\authorrunning{F. Author et al.}` for the page header. Your paper has `\titlerunning` but **no `\authorrunning`**. |

### Important — Formatting Deviations

| Line | Issue | Detail |
|------|-------|--------|
| 11–12 | **`\usepackage{caption}` and `\usepackage{subcaption}`** | The `llncs` class has its own caption formatting. Loading `caption` can override LNCS caption styles (font size, spacing). Consider removing unless strictly needed. |
| 29 | **`colorlinks=true`** | LNCS print proceedings require **black text** — colored links will appear in the print version. The template comments out `hyperref` color options. For submission, use `hidelinks` or remove `colorlinks`. |
| 9 | **`\usepackage{xcolor}`** + line 22 `\usepackage{color}` | Both `xcolor` and `color` are loaded — `xcolor` supersedes `color`. Remove line 22 (`\usepackage{color}`). |
| 685 | **`\newpage` before bibliography** | Not in the template. LNCS layout handles page breaks automatically. Remove unless there's a specific reason. |
| 16 | **`\usepackage{tikz}` + line 24 `\usetikzlibrary{...}`** | TikZ is loaded with multiple libraries but doesn't appear to be used anywhere in the paper. Remove to reduce compilation overhead. |

### Minor — Style Consistency

| Line | Issue | Detail |
|------|-------|--------|
| 72 | Template uses `\caption{...}\label{...}` then `\begin{tabular}` | Your paper generally follows this, but some tables have inconsistent ordering (e.g., `\label` after `\end{tabular}` at lines 383, 419). LNCS convention: `\caption` + `\label` **above** the table, before `\begin{tabular}`. |
| 10 | **`\usepackage{enumitem}`** | The `llncs` class defines its own list environments. `enumitem` may conflict with their spacing. Test carefully. |
| 13 | **`\usepackage{float}` for `[H]` placement** | Heavy use of `[H]` overrides LaTeX's float placement algorithm. LNCS papers typically let floats position naturally. Consider relaxing to `[htbp]` where possible. |
| 47 | **`\institute` missing email** | Template shows email inside `\institute{}`. Your paper has the email commented out (line 48). Add it back for completeness. |
