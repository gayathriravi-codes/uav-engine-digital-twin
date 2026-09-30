/**
 * src/data/apiSource.js
 * Client for the FastAPI backend with WebSocket stream,
 * REST endpoints, and automatic fallback to SimSource.
 */

import { API_BASE_URL } from "../config";
import { normalizeTick } from "./normalizeTick";

export class ApiSourceClient {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.wsUrl = this.baseUrl.replace(/^http/, "ws");
    this.ws = null;
    this.isConnected = false;
  }

  /**
   * Health check to test backend reachability
   */
  async checkHealth() {
    try {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: controller.signal,
      });
      clearTimeout(id);
      if (!res.ok) return false;
      const data = await res.json();
      return Boolean(data.faults);
    } catch {
      return false;
    }
  }

  /**
   * Fetch complete flight for replay / scrubber
   */
  async fetchFlight({ fault = "none", severity = 1.0, seed = 42 } = {}) {
    const params = new URLSearchParams({
      fault,
      severity: String(severity),
      seed: String(seed),
    });
    const res = await fetch(`${this.baseUrl}/flight?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to fetch flight: ${res.statusText}`);
    }
    const rawList = await res.json();
    let prev = null;
    return rawList.map((raw) => {
      const norm = normalizeTick(raw, prev);
      prev = norm;
      return norm;
    });
  }

  /**
   * Query counterfactual flight for What-If scenario
   */
  async fetchWhatIf({ fault, severity = 1.0, seed = 42 } = {}) {
    const res = await fetch(`${this.baseUrl}/whatif`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fault, severity, seed }),
    });
    if (!res.ok) {
      throw new Error(`WhatIf request failed: ${res.statusText}`);
    }
    const rawList = await res.json();
    let prev = null;
    return rawList.map((raw) => {
      const norm = normalizeTick(raw, prev);
      prev = norm;
      return norm;
    });
  }

  /**
   * Post operator confirmation decision
   */
  async postConfirm({ session = "default", t, decision }) {
    const res = await fetch(`${this.baseUrl}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session, t, decision }),
    });
    if (!res.ok) {
      throw new Error(`Confirmation failed: ${res.statusText}`);
    }
    return await res.json();
  }

  /**
   * Connect to WebSocket telemetry stream
   */
  connectStream({
    fault = "none",
    severity = 1.0,
    seed = 42,
    tickSeconds = 1.0,
    session = "default",
    onTick,
    onError,
    onClose,
    onStatusChange,
  }) {
    if (this.ws) {
      this.disconnectStream();
    }

    onStatusChange?.("connecting");

    const params = new URLSearchParams({
      fault,
      severity: String(severity),
      seed: String(seed),
      tick_seconds: String(tickSeconds),
      session,
    });

    const url = `${this.wsUrl}/stream?${params}`;

    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      onStatusChange?.("unreachable");
      onError?.(err);
      return;
    }

    let prevTick = null;

    this.ws.onopen = () => {
      this.isConnected = true;
      onStatusChange?.("connected");
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "end") {
          onClose?.();
          return;
        }
        if (msg.type === "error") {
          onError?.(new Error(msg.detail || "Stream error"));
          return;
        }
        const norm = normalizeTick(msg, prevTick);
        prevTick = norm;
        onTick?.(norm);
      } catch (err) {
        console.warn("Failed to parse incoming tick", err);
      }
    };

    this.ws.onerror = (err) => {
      this.isConnected = false;
      onStatusChange?.("unreachable");
      onError?.(err);
    };

    this.ws.onclose = () => {
      this.isConnected = false;
      onStatusChange?.("unreachable");
      onClose?.();
    };
  }

  disconnectStream() {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
      this.isConnected = false;
    }
  }
}

export const apiSource = new ApiSourceClient();
