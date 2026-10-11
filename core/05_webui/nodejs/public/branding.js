/**
 * Per-pipeline branding.
 *
 * All three departments (chips / fg / fg2) are served by this same set of static
 * files, so nothing in the HTML can be department-specific. This fetches the
 * identity of whichever pipeline is serving the page and fills in every element
 * marked with a data-brand-* attribute:
 *
 *   data-brand-logo     <img>  emblem src
 *   data-brand-name     text   short department name  (e.g. "GAD")
 *   data-brand-sub      text   subtitle               (e.g. "Governance Assistant")
 *   data-brand-org      text   full department name
 *   data-brand-favicon  <link> tab icon
 *
 * Runs before login so the sign-in card is already branded. Needs no auth token:
 * the endpoint returns display strings only.
 */
'use strict';

(async function applyBranding() {
  const FALLBACK = {
    shortLabel: 'Document Assistant',
    subtitle: '',
    org: 'Government of Chhattisgarh',
    label: 'Document Assistant',
    logo: '',
  };

  let brand = FALLBACK;
  try {
    const base = (typeof window !== 'undefined' && window.__APP_BASE__) || '';
    const res = await fetch(`${base}/api/branding`, { credentials: 'same-origin' });
    if (res.ok) brand = Object.assign({}, FALLBACK, await res.json());
  } catch (_) {
    // Offline or endpoint missing — keep the neutral fallback rather than
    // showing another department's name.
  }

  const setText = (attr, value) => {
    if (!value) return;
    document.querySelectorAll(`[${attr}]`).forEach((el) => { el.textContent = value; });
  };

  setText('data-brand-name', brand.shortLabel);   // compact: "GAD"
  setText('data-brand-label', brand.label);        // full: "GAD Governance Assistant"
  setText('data-brand-sub', brand.subtitle);
  setText('data-brand-org', brand.org);

  if (brand.logo) {
    document.querySelectorAll('[data-brand-logo]').forEach((img) => {
      img.src = brand.logo;
      img.alt = `${brand.shortLabel} emblem`;
      img.hidden = false;
    });
    const icon = document.querySelector('[data-brand-favicon]');
    if (icon) icon.href = brand.logo;
  }

  // Page title: "<Department>" on the main page, "<Department> • <Page>" elsewhere.
  // The sub-page name comes from an explicit <meta name="page-name">; deriving it
  // from the existing document.title produced "GAD Governance Assistant •
  // Document Assistant" on the page whose title was already the department.
  const pageName = document.querySelector('meta[name="page-name"]')?.content?.trim();
  document.title = pageName ? `${brand.label} • ${pageName}` : brand.label;

  document.documentElement.dataset.pipeline = brand.pipeline || '';
})();
