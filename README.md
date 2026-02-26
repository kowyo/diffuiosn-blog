# UINUX Blog

A calm, writing-first blogging system built with Astro.

## What it is

UINUX Blog is a static blog system designed for long-form writing. It prioritizes typography, whitespace, and readability over visual complexity. It uses Markdown for content, file-based routing, and ships minimal client-side JavaScript (search only).

## What it is not

- Not a theme. There are no color presets, dark mode toggles, or visual variants.
- Not a CMS. There is no admin panel, no database, no integrations.
- Not a marketing tool. There are no newsletter signups, analytics, or growth hacks.
- Not a portfolio. There are no image galleries, project showcases, or landing pages.

## Who it is for

Writers who want a quiet place to publish. People who value clarity over features. Anyone who prefers to write in Markdown and deploy a static site.

## Who it is not for

Anyone who needs a visual builder, dynamic content, user accounts, or comments. If you need those things, this is not the right tool.

## Getting started

### Prerequisites

- Node.js 18 or later

### Install

```
pnpm install
```

### Start writing

1. Create a new `.md` or `.mdx` file in `src/content/posts/`
2. Add frontmatter:

```md
---
title: "Your Post Title"
description: "A short description of the post."
date: 2026-02-07
---

Your content here.
```

3. Run the dev server:

```
pnpm dev
```

4. Open `http://localhost:4321`

That is it. You are writing.

### Google Search Console

To enable Google Search Console verification, create a `.env` file:

```
PUBLIC_GOOGLE_SITE_VERIFICATION=your-verification-code
```

Get the code from Google Search Console > Settings > Ownership verification > HTML tag.

See `.env.example` for reference.

### Build for production

```
pnpm build
```

Output is in the `dist/` directory. Deploy it anywhere that serves static files.

## Structure

```
src/
  components/
    Layout.astro      Page shell, meta, fonts
    Nav.astro          Site navigation
    Footer.astro       Site footer
    Heading.astro      Typographic heading
    Article.astro      Post wrapper with metadata
    Prose.astro        Markdown content styling
  content/
    config.ts          Content collection schema
    posts/             Markdown files go here
  pages/
    index.astro              Home (post list)
    search.astro             Search page (client-side)
    search-index.json.ts     Static search index (built at build time)
    about.astro              About page
    rss.xml.ts               RSS feed
    posts/[...slug].astro    Individual post pages
  styles/
    global.css         Design tokens and base styles
```

## Design decisions

- **Serif body text** (Newsreader): optimized for sustained reading
- **Sans-serif headings** (Inter): clear hierarchy without competing with body text
- **640px content width**: prevents eye fatigue on long lines
- **Minimal client-side JS**: only used for search; all other pages are static HTML and CSS
- **No dark mode**: one mode, one experience, no maintenance surface
- **No component library**: six components, no more

## SEO

Built-in, zero-dependency SEO on every page:

- **Canonical URLs** on all pages
- **Open Graph meta** (og:title, og:description, og:url, og:type, og:site_name, og:image)
- **Twitter Card meta** (summary or summary_large_image)
- **JSON-LD structured data**:
  - `WebSite` on the home page
  - `BlogPosting` on each post (headline, datePublished, author, publisher)
  - `AboutPage` on the about page
- **Google Search Console** verification via environment variable
- **RSS feed** at `/rss.xml`

No third-party SEO libraries. No sitemap generator. No tracking.

## Search

Client-side search powered by a static JSON index built at build time.

- No external services or APIs
- No dependencies
- Index includes title, description, and full post body
- Debounced input with instant results
- Supports `?q=` query parameter for direct linking
- Available at `/search`

## Content model

Posts require three frontmatter fields:

| Field         | Type   | Required |
|---------------|--------|----------|
| `title`       | string | yes      |
| `description` | string | yes      |
| `date`        | date   | yes      |

No tags. No categories. No authors. Write and publish.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `PUBLIC_GOOGLE_SITE_VERIFICATION` | no | Google Search Console verification code |

## License

MIT
