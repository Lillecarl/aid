// Speech to text through aid web: the microphone streams to /api/transcribe, and what the server hears comes back.

export interface Heard {
  text: string;
  final: boolean;
}

// About 100 ms of audio per WebSocket frame: one frame per 128-sample quantum would be hundreds a second.
const FRAME_SECONDS = 0.1;

export class Dictation {
  private socket: WebSocket | null = null;
  private context: AudioContext | null = null;
  private media: MediaStream | null = null;
  private pending: Float32Array[] = [];
  private pendingLength = 0;

  constructor(private readonly onheard: (heard: Heard) => void) {}

  /** Ask for the microphone and start streaming. Rejects if the browser refuses the microphone. */
  async start(): Promise<void> {
    this.media = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    this.context = new AudioContext();
    await this.context.audioWorklet.addModule(new URL("./mic-worklet.js", import.meta.url));
    const rate = this.context.sampleRate;
    const frame = Math.round(rate * FRAME_SECONDS);

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/api/transcribe`);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.onmessage = (event: MessageEvent<string>) => {
      const data = JSON.parse(event.data) as Partial<Heard> & { done?: boolean };
      if (typeof data.text === "string") this.onheard({ text: data.text, final: data.final === true });
    };
    await new Promise<void>((resolve, reject) => {
      socket.onopen = () => resolve();
      socket.onerror = () => reject(new Error("could not reach speech to text"));
    });
    socket.send(JSON.stringify({ rate }));

    const node = new AudioWorkletNode(this.context, "aid-mic");
    node.port.onmessage = (event: MessageEvent<Float32Array>) => {
      this.pending.push(event.data);
      this.pendingLength += event.data.length;
      if (this.pendingLength >= frame) this.flush();
    };
    this.context.createMediaStreamSource(this.media).connect(node);
  }

  private flush(): void {
    if (this.socket?.readyState !== WebSocket.OPEN || this.pendingLength === 0) return;
    const samples = new Float32Array(this.pendingLength);
    let offset = 0;
    for (const chunk of this.pending) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    this.pending = [];
    this.pendingLength = 0;
    // Float32Array is the platform's byte order, little-endian on every browser that runs this page.
    this.socket.send(samples.buffer);
  }

  /** Stop listening. Resolves once the server has sent its last words. */
  async stop(): Promise<void> {
    this.flush();
    for (const track of this.media?.getTracks() ?? []) track.stop();
    await this.context?.close();
    const socket = this.socket;
    if (socket?.readyState !== WebSocket.OPEN) return;
    await new Promise<void>((resolve) => {
      socket.addEventListener("close", () => resolve());
      socket.send(JSON.stringify({ end: true }));
    });
  }
}
