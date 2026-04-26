// scripts/sync_obsidian_to_blog.cjs
// 用于将 Obsidian 日记同步到 Astro 博客的 posts 目录
// 使用方法：node scripts/sync_obsidian_to_blog.cjs

const fs = require('fs');
const path = require('path');
const fse = require('fs-extra');
const matter = require('gray-matter');

// === 配置区 ===
// 默认 Obsidian 日记目录（WSL2 下 Windows 路径挂载，/mnt/e/...）
const DEFAULT_OBSIDIAN_DIR = '/mnt/e/Obsidian note/Note/Academic';
// Astro 博客 posts 目录（相对本项目根目录）
const BLOG_POSTS_DIR = path.join(__dirname, '../src/content/posts');

// === 同步逻辑 ===
function getFileDate(filePath) {
  const stat = fs.statSync(filePath);
  const hasValidBirthtime = stat.birthtime instanceof Date
    && !Number.isNaN(stat.birthtime.getTime())
    && stat.birthtime.getTime() > 0
    && stat.birthtime.getUTCFullYear() > 1970;

  return hasValidBirthtime ? stat.birthtime : stat.mtime;
}

function normalizeMathBlocks(content) {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const normalized = [];
  let inBlockMath = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed === '$$' || trimmed === '$') {
      if (!inBlockMath) {
        if (normalized.length > 0 && normalized[normalized.length - 1] !== '') {
          normalized.push('');
        }
        normalized.push('$$');
        inBlockMath = true;
      } else {
        normalized.push('$$');
        normalized.push('');
        inBlockMath = false;
      }
      continue;
    }

    normalized.push(line);
  }

  while (normalized.length > 0 && normalized[normalized.length - 1] === '') {
    normalized.pop();
  }

  return `${normalized.join('\n')}\n`;
}

function applyOutsideFencedCodeBlocks(content, transform) {
  const parts = content.split(/(```[\s\S]*?```)/g);
  return parts
    .map((part, index) => (index % 2 === 1 ? part : transform(part)))
    .join('');
}

function normalizeLeadingTabs(content) {
  return applyOutsideFencedCodeBlocks(content, (segment) => segment
    // 将行首 Tab 统一为两个空格，避免 Markdown 把列表中的公式误识别为代码块。
    .replace(/^(\t+)/gm, (tabs) => '  '.repeat(tabs.length)));
}

function normalizeMathSegment(segment) {
  let normalized = segment
    .replace(/’/g, "'")
    .replace(/‘/g, "'")
    .replace(/–/g, '-')
    .replace(/—/g, '--');

  const hasMathEnvironment = /\\begin\{(?:aligned|align\*?|split|matrix|bmatrix|pmatrix|vmatrix|Vmatrix|cases|array|gather\*?)\}/.test(normalized);

  if (!hasMathEnvironment) {
    normalized = normalized
      .replace(/^\s*\\\\\s*$/gm, '')
      .replace(/\\\\\s*$/gm, '');
  }

  normalized = normalized
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return normalized;
}

function normalizeMathContent(content) {
  let normalized = applyOutsideFencedCodeBlocks(content, (segment) => segment.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathBody, offset, whole) => {
    const prevChar = offset > 0 ? whole[offset - 1] : '\n';
    const nextIndex = offset + match.length;
    const nextChar = nextIndex < whole.length ? whole[nextIndex] : '\n';
    const isStandaloneBlock = prevChar === '\n' && (nextChar === '\n' || typeof nextChar === 'undefined');
    const cleaned = normalizeMathSegment(mathBody);

    if (isStandaloneBlock) {
      return `$$\n${cleaned}\n$$`;
    }

    return `$${cleaned.replace(/\n+/g, ' ')}$`;
  }));

  normalized = applyOutsideFencedCodeBlocks(normalized, (segment) => segment.replace(/(?<!\$)\$(\\begin\{equation\*?\}[\s\S]*?\\end\{equation\*?\})\$(?!\$)/g, (match, mathBody) => {
    const cleaned = normalizeMathSegment(mathBody);
    return `$$\n${cleaned}\n$$`;
  }));

  normalized = normalized.replace(/(?<!\$)\$(?!\$)([^\n$]+?)(?<!\$)\$(?!\$)/g, (match, mathBody) => {
    const cleaned = normalizeMathSegment(mathBody);
    return `$${cleaned}$`;
  });

  return normalized;
}

function normalizeDetachedBlockMathDelimiters(content) {
  return applyOutsideFencedCodeBlocks(content, (segment) => segment
    // 将“文字$$”拆成“文字\n$$”，避免块公式起始符粘连正文导致后续误解析。
    .replace(/([^\s\n])\s*\$\$(?=\s*(?:\n|$))/g, '$1\n$$')
    // 将“$$文字”拆成“$$\n文字”，避免块公式结束符与正文粘连。
    .replace(/(^|\n)\s*\$\$\s*([^\s\n])/g, '$1$$\n$2'));
}

function slugifySegment(value) {
  const normalized = String(value)
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .trim()
    .toLowerCase();

  const slug = normalized
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
    .replace(/^-+|-+$/g, '');

  return slug || normalized || 'post';
}

function buildObsidianWikiHref(target) {
  const rawTarget = target.trim();
  if (!rawTarget) {
    return null;
  }

  if (rawTarget.startsWith('#')) {
    const heading = rawTarget.slice(1).trim();
    if (!heading) {
      return null;
    }
    return `#${slugifySegment(heading)}`;
  }

  let notePath = rawTarget;
  let anchor = '';

  if (rawTarget.includes('#')) {
    const index = rawTarget.indexOf('#');
    notePath = rawTarget.slice(0, index).trim();
    anchor = rawTarget.slice(index + 1).trim();
  }

  // Obsidian 支持 [[Folder/Note]]，博客使用最终文件名作为 slug。
  const noteName = path.basename(notePath.replace(/\.md$/i, '').trim());
  if (!noteName) {
    return null;
  }

  const postSlug = slugifySegment(noteName);
  const anchorPart = anchor ? `#${slugifySegment(anchor)}` : '';
  return `/posts/${postSlug}/${anchorPart}`;
}

function convertObsidianWikiLinks(segment) {
  return segment.replace(/(?<!!)\[\[([^\]]+)\]\]/g, (match, rawInner) => {
    const inner = rawInner.trim();
    if (!inner) {
      return match;
    }

    const [rawTarget, rawAlias] = inner.split('|');
    const target = (rawTarget || '').trim();
    const alias = rawAlias ? rawAlias.trim() : '';
    const href = buildObsidianWikiHref(target);

    if (!href) {
      return match;
    }

    const fallbackText = target.startsWith('#')
      ? target.slice(1).trim()
      : path.basename(target.replace(/\.md$/i, '').split('#')[0].trim());
    const linkText = alias || fallbackText || target;
    return `[${linkText}](${href})`;
  });
}

function normalizeObsidianLinks(content) {
  // 不处理 fenced code block 内部，避免把示例语法误替换为链接。
  return applyOutsideFencedCodeBlocks(content, convertObsidianWikiLinks);
}

function applyOutsideMarkdownSensitiveSegments(content, transform) {
  const pattern = /(```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|(?<!\$)\$[^\n$]+\$|!\[[^\]]*\]\([^\)]*\)|\[[^\]]*\]\([^\)]*\))/g;
  const parts = content.split(pattern);
  return parts
    .map((part, index) => {
      if (!part) return part;
      return index % 2 === 1 ? part : transform(part);
    })
    .join('');
}

function normalizeCjkSpacing(content) {
  return applyOutsideMarkdownSensitiveSegments(content, (segment) => segment
    // 中文与英文字母/数字之间自动补空格。
    .replace(/([\u4e00-\u9fff])([A-Za-z0-9])/g, '$1 $2')
    .replace(/([A-Za-z0-9])([\u4e00-\u9fff])/g, '$1 $2')
    // 中文与常见西文符号（如 @#&）之间补空格，提升可读性。
    .replace(/([\u4e00-\u9fff])([@#&])/g, '$1 $2')
    .replace(/([@#&])([\u4e00-\u9fff])/g, '$1 $2')
    // 去掉中文收尾标点前的多余空格，避免“文字 ，”这类排版。
    .replace(/\s+([，。！？；：、）】》」』])/g, '$1')
    // 去掉中文开头标点后的多余空格，避免“（ 说明）”这类排版。
    .replace(/([（【《「『])\s+/g, '$1')
    // 收敛连续空白字符，保持段落观感稳定。
    .replace(/[ \t]{2,}/g, ' '));
}

function fixFrontmatter(raw, filePath) {
  const { data, content } = matter(raw);
  const normalizedContent = normalizeCjkSpacing(normalizeObsidianLinks(
    normalizeMathContent(normalizeMathBlocks(normalizeDetachedBlockMathDelimiters(normalizeLeadingTabs(content)))),
  ));
  let changed = false;
  if (!data.title) {
    data.title = path.basename(filePath, '.md');
    changed = true;
  }
  if (!data.date) {
    data.date = getFileDate(filePath);
    changed = true;
  }
  if (!data.tags) {
    data.tags = [];
    changed = true;
  } else if (typeof data.tags === 'string') {
    data.tags = [data.tags];
    changed = true;
  }
  if (typeof data.draft === 'undefined') {
    data.draft = false;
    changed = true;
  }
  if (normalizedContent !== content) {
    changed = true;
  }
  if (changed) {
    return matter.stringify(normalizedContent, data);
  } else {
    return raw;
  }
}

function parseArgs(argv) {
  let inputPath;
  let dryRun = false;

  for (const arg of argv) {
    if (arg === '--dry-run') {
      dryRun = true;
      continue;
    }

    if (!inputPath) {
      inputPath = arg;
    }
  }

  return { inputPath, dryRun };
}

function syncNotes(inputPath, options = {}) {
  const { dryRun = false } = options;

  if (!fs.existsSync(BLOG_POSTS_DIR)) {
    fse.mkdirpSync(BLOG_POSTS_DIR);
  }

  let files = [];
  let baseDir = '';
  let syncMode = '';

  if (!inputPath) {
    // 未指定参数，默认同步整个目录
    baseDir = DEFAULT_OBSIDIAN_DIR;
    syncMode = 'all';
    if (!fs.existsSync(baseDir)) {
      console.error('Obsidian 源目录不存在:', baseDir);
      process.exit(1);
    }
    files = fs.readdirSync(baseDir)
      .filter(f => f.endsWith('.md'))
      .map(f => path.join(baseDir, f));
  } else {
    const stat = fs.statSync(inputPath);
    if (stat.isDirectory()) {
      baseDir = inputPath;
      syncMode = 'directory';
      files = fs.readdirSync(baseDir)
        .filter(f => f.endsWith('.md'))
        .map(f => path.join(baseDir, f));
    } else if (stat.isFile() && inputPath.endsWith('.md')) {
      baseDir = path.dirname(inputPath);
      syncMode = 'file';
      files = [inputPath];
    } else {
      console.error('输入路径不是有效的 Markdown 文件或目录:', inputPath);
      process.exit(1);
    }
  }

  if (syncMode === 'all') {
    console.log(`同步模式: 全量同步`);
    console.log(`源目录: ${baseDir}`);
  } else if (syncMode === 'directory') {
    console.log(`同步模式: 目录同步`);
    console.log(`源目录: ${baseDir}`);
  } else if (syncMode === 'file') {
    console.log(`同步模式: 单文件同步`);
    console.log(`源文件: ${files[0]}`);
  }
  console.log(`目标目录: ${BLOG_POSTS_DIR}`);
  console.log(`执行模式: ${dryRun ? '预览，不写入文件' : '写入同步'}`);

  let syncedCount = 0;
  let skippedCount = 0;
  for (const src of files) {
    const file = path.basename(src);
    const dest = path.join(BLOG_POSTS_DIR, file);
    const raw = fs.readFileSync(src, 'utf-8');
    const fixed = fixFrontmatter(raw, src);

    if (fs.existsSync(dest)) {
      const current = fs.readFileSync(dest, 'utf-8');
      if (current === fixed) {
        console.log(`跳过未变化文件: ${file}`);
        skippedCount++;
        continue;
      }
    }

    if (dryRun) {
      console.log(`将同步文件: ${file}`);
    } else {
      fs.writeFileSync(dest, fixed, 'utf-8');
      console.log(`已同步文件: ${file}`);
    }
    syncedCount++;
  }

  if (dryRun) {
    console.log(`预览完成: ${syncedCount} 个文件将被同步，${skippedCount} 个文件无变化`);
  } else {
    console.log(`同步完成: ${syncedCount} 个文件已写入，${skippedCount} 个文件无变化`);
  }
}

const { inputPath, dryRun } = parseArgs(process.argv.slice(2));
syncNotes(inputPath, { dryRun });
