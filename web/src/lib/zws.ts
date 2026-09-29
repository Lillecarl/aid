// ZWS 2.0 (zeromq over WebSocket) to aid web's relays (aid/web/zws.py): each WebSocket message is one frame, a
// flags byte and then the body; a multipart message is several, MORE on all but the last.

const SUBPROTOCOL = "ZWS2.0";
const MORE = 0x01;
const COMMAND = 0x02;
// A ZMTP 3.1 PING command frame: the flags, "\x04PING", a TTL of 0 (none) and no context.
const PING = new Uint8Array([COMMAND, 4, 0x50, 0x49, 0x4e, 0x47, 0, 0]);
const HEARTBEAT_MS = 15_000;
const RETRY_MIN_MS = 500;
const RETRY_MAX_MS = 10_000;

export interface Handlers {
  /** A whole multipart message from the daemon. */
  message(parts: Uint8Array[]): void;
  /** The WebSocket is open: send what a new connection needs. */
  open(): void;
  /** The WebSocket closed; it reconnects while started. `opened` is false when it never opened at all, as when the
   * login has gone and aid web refuses the upgrade. */
  close(opened: boolean): void;
}

function frame(flags: number, body: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(body.length + 1);
  out[0] = flags;
  out.set(body, 1);
  return out;
}

/** One of aid web's ZWS relays, reconnecting with backoff while started. A PING every HEARTBEAT_MS tells a dead
 * connection from a quiet one: a PING still unanswered at the next one closes it. Not a clock: a hidden tab's
 * timers run up to a minute late, and a late timer must not close a live connection. */
export class ZwsSocket {
  private ws: WebSocket | null = null;
  private parts: Uint8Array[] = [];
  private started = false;
  private retryMs = RETRY_MIN_MS;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  private beatTimer: ReturnType<typeof setInterval> | undefined;
  private unanswered = false;

  constructor(
    private readonly path: string,
    private readonly handlers: Handlers,
  ) {}

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    this.connect();
  }

  stop(): void {
    this.started = false;
    clearTimeout(this.retryTimer);
    this.drop();
  }

  send(parts: Uint8Array[]): void {
    const ws = this.ws;
    if (ws === null || ws.readyState !== WebSocket.OPEN)
      throw new Error("not connected to aid web");
    parts.forEach((part, i) =>
      ws.send(frame(i < parts.length - 1 ? MORE : 0, part)),
    );
  }

  private connect(): void {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${scheme}://${location.host}${this.path}`,
      SUBPROTOCOL,
    );
    ws.binaryType = "arraybuffer";
    let opened = false;
    ws.onopen = () => {
      opened = true;
      this.retryMs = RETRY_MIN_MS;
      this.unanswered = false;
      this.beatTimer = setInterval(() => this.beat(), HEARTBEAT_MS);
      this.handlers.open();
    };
    ws.onmessage = (event: MessageEvent<unknown>) => {
      this.unanswered = false;
      if (!(event.data instanceof ArrayBuffer) || event.data.byteLength === 0)
        return;
      const data = new Uint8Array(event.data);
      const flags = data[0] ?? 0;
      if (flags & COMMAND) return; // A PONG: hearing it was the point.
      this.parts.push(data.subarray(1));
      if (!(flags & MORE)) {
        const whole = this.parts;
        this.parts = [];
        this.handlers.message(whole);
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.drop();
      this.handlers.close(opened);
      if (!this.started) return;
      this.retryTimer = setTimeout(() => this.connect(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, RETRY_MAX_MS);
    };
    this.ws = ws;
  }

  private drop(): void {
    clearInterval(this.beatTimer);
    const ws = this.ws;
    this.ws = null;
    this.parts = [];
    ws?.close();
  }

  private beat(): void {
    if (this.unanswered) {
      this.ws?.close(); // onclose reconnects.
      return;
    }
    this.unanswered = true;
    this.ws?.send(PING);
  }
}
