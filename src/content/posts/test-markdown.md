---
title: Markdown Feature Test
subtitle: A comprehensive reference for all supported syntax
date: 2015-02-28
tags: [test, markdown, reference]
---

This post serves as a living reference for all Markdown features supported by this site. Use it to verify rendering after configuration changes.

---

## Typography

Regular paragraph text. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

**Bold text** and *italic text* and ***bold italic text***.

~~Strikethrough~~ and `inline code` and a [link to markdowntutorial.com](http://markdowntutorial.com/).

Superscript: 10^2 = 100. Subscript is not standard Markdown.

---

## Headings

# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6

---

## Lists

### Unordered

- Item A
- Item B
  - Nested B1
  - Nested B2
    - Deep nested
- Item C

### Ordered

1. First
2. Second
3. Third
   1. Sub-step 3.1
   2. Sub-step 3.2

### Task List

- [x] Migrate from Jekyll to Astro
- [x] Add Matrix rain background
- [x] Configure Tailwind Typography
- [ ] Add RSS feed
- [ ] Add pagination

---

## Code

### Inline

Use `npm run dev` to start the development server. The config lives in `astro.config.mjs`.

### Fenced block (no language)

```
var foo = function(x) {
  return x + 5;
};
foo(3);
```

### JavaScript

```javascript
// Astro content collection example
import { getCollection } from "astro:content";

const posts = (await getCollection("posts")).sort(
  (a, b) => b.data.date.getTime() - a.data.date.getTime()
);
```

### TypeScript

```typescript
interface Post {
  title: string;
  date: Date;
  tags?: string[];
}

function formatDate(date: Date): string {
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}
```

### CSS / Tailwind

```css
/* Custom alert styles */
.markdown-alert-note {
  border-color: #3b82f6;
  background-color: rgba(59, 130, 246, 0.1);
}
```

### Shell

```sh
npm install
npm run dev
npm run build
```

---

## Tables

| Feature        | Jekyll | Astro     |
| :------------- | :----- | :-------- |
| Templating     | Liquid | Astro JSX |
| Styling        | SASS   | Tailwind  |
| Build speed    | Slow   | Fast      |
| TypeScript     | ✗      | ✓         |
| Content Layer  | ✗      | ✓         |

---

## Images

Standard image:

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg)

Image with title attribute (hover tooltip):

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg "A delicious crepe")

---

## Blockquotes

Plain blockquote:

> The best way to predict the future is to invent it.
> — Alan Kay

Nested blockquote:

> Outer quote.
> > Inner nested quote.

---

## GitHub Alerts

> [!NOTE]
> Useful information that users should know, even when skimming.

> [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.

---

## Horizontal Rules

Above this line.

---

Below this line.

***

Also a horizontal rule.

---

## Links

- [External link](https://astro.build)
- [External link with title](https://tailwindcss.com "Tailwind CSS docs")
- [Root-relative link to Blog page](/blog)
- [Root-relative link to About page](/about)

---

## Escaping

Literal asterisks: \*not italic\*

Literal backtick: \`not code\`

Literal bracket: \[not a link\]

---

## Long-form Paragraph

This section tests prose readability — line length, spacing, and overall legibility against the dark backdrop of this site. A well-configured `prose prose-invert` class from Tailwind Typography should make this comfortable to read without any additional tweaking.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque vehicula enim vitae sem tincidunt, at consequat nisi ultrices. Donec aliquam libero ac felis fermentum, at malesuada nisl efficitur. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae.
