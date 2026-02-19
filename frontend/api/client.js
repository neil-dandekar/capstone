// @ts-check

/**
 * @typedef {import("./types").CbllmRunRequest} CbllmRunRequest
 * @typedef {import("./types").CbllmRunSuccess} CbllmRunSuccess
 * @typedef {import("./types").CbllmRunErrorBody} CbllmRunErrorBody
 */

(function initCbllmApi(global) {
  const DEFAULT_BASE_URL = "http://localhost:8000";
  const LOOPBACK_HOSTS = ["localhost", "127.0.0.1"];

  function readApiOverrideFromQuery() {
    try {
      const qs = global.location && global.location.search ? global.location.search : "";
      const params = new URLSearchParams(qs);
      const api = (params.get("api") || "").trim();
      if (!api) return null;

      // Allow either "https://host" or "https://host/" etc.
      // Basic sanity check (avoid weird values)
      if (!/^https?:\/\/[^ ]+$/i.test(api)) return null;
      return api.replace(/\/+$/, "");
    } catch {
      return null;
    }
  }

  /**
   * @param {{ baseUrl?: string }} [options]
   * @returns {string[]}
   */
  function resolveBaseUrls(options) {
    // 1) explicit call override
    if (options && options.baseUrl) {
      return [String(options.baseUrl).replace(/\/+$/, "")];
    }

    // 2) URL query override: ?api=https://your-backend.com
    const queryOverride = readApiOverrideFromQuery();
    if (queryOverride) {
      return [queryOverride];
    }

    // 3) dev fallbacks
    const currentHost = global.location && global.location.hostname;
    const preferred = [];

    if (LOOPBACK_HOSTS.includes(currentHost)) {
      preferred.push(`http://${currentHost}:8000`);
    }
    LOOPBACK_HOSTS.forEach((host) => {
      preferred.push(`http://${host}:8000`);
    });
    preferred.push(DEFAULT_BASE_URL);

    return Array.from(new Set(preferred.map((u) => u.replace(/\/+$/, ""))));
  }

  /**
   * @param {string} baseUrl
   * @param {CbllmRunRequest} payload
   * @returns {Promise<CbllmRunSuccess>}
   */
  async function postRunToBaseUrl(baseUrl, payload) {
    const url = `${baseUrl.replace(/\/+$/, "")}/api/v1/run`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await res.text();
    /** @type {unknown} */
    let body = null;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = { error: { code: "INVALID_JSON", message: text } };
      }
    }

    if (!res.ok) {
      /** @type {CbllmRunErrorBody | any} */
      const errBody =
        body || {
          error: { code: `HTTP_${res.status}`, message: res.statusText },
          request_id: payload.request_id || null,
        };

      const err = new Error(
        errBody?.error?.message || `Request failed (${res.status})`,
      );
      // @ts-ignore custom error fields for UI rendering
      err.status = res.status;
      // @ts-ignore
      err.code = errBody?.error?.code || `HTTP_${res.status}`;
      // @ts-ignore
      err.details = errBody?.error?.details || [];
      // @ts-ignore
      err.request_id = errBody?.request_id || payload.request_id || null;
      // @ts-ignore
      err.baseUrl = baseUrl;
      throw err;
    }

    return /** @type {CbllmRunSuccess} */ (body);
  }

  /**
   * @param {CbllmRunRequest} payload
   * @param {{ baseUrl?: string }} [options]
   * @returns {Promise<CbllmRunSuccess>}
   */
  async function postRun(payload, options) {
    const baseUrls = resolveBaseUrls(options);
    /** @type {unknown} */
    let lastError = null;

    for (let i = 0; i < baseUrls.length; i++) {
      const baseUrl = baseUrls[i];
      try {
        return await postRunToBaseUrl(baseUrl, payload);
      } catch (err) {
        // Retry only if this is a network-level failure (fetch throws TypeError).
        if (err instanceof TypeError) {
          lastError = err;
          continue;
        }
        throw err;
      }
    }

    const err = new Error("Unable to reach backend (tried configured base URLs).");
    // @ts-ignore
    err.code = "BACKEND_UNREACHABLE";
    // @ts-ignore
    err.details = [
      {
        tried: baseUrls,
        message: lastError instanceof Error ? lastError.message : "Network error",
      },
    ];
    // @ts-ignore
    err.request_id = payload.request_id || null;
    throw err;
  }

  global.CbllmApi = {
    postRun,
    DEFAULT_BASE_URL,
    resolveBaseUrls, // handy for debugging
  };
})(window);
