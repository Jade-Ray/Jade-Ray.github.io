export function slugifyTag(tag: string) {
  const normalized = String(tag)
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase();

  const slug = normalized
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return slug || normalized || "tag";
}

export function getTagHref(tag: string) {
  return `/tags/${slugifyTag(tag)}`;
}

export function getTagPageHref(tag: string, page: number) {
  const base = getTagHref(tag);
  return page <= 1 ? base : `${base}/${page}`;
}