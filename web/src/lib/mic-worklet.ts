// The microphone's first channel, handed to the page one render quantum (128 samples) at a time. Vite bundles it on
// its own (`?worker&url` in dictation.ts), and the page loads it with `audioWorklet.addModule`.

// The AudioWorklet global scope. TypeScript's lib for it conflicts with the DOM lib the rest of the UI uses, so the
// three names this file needs are declared here.
declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  abstract process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean;
}
declare function registerProcessor(name: string, processor: new () => AudioWorkletProcessor): void;

class Mic extends AudioWorkletProcessor {
  process(inputs: Float32Array[][]): boolean {
    const channel = inputs[0]?.[0];
    if (channel) this.port.postMessage(channel.slice());
    return true;
  }
}

registerProcessor("aid-mic", Mic);
