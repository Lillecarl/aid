// The microphone's first channel, handed to the page one render quantum (128 samples) at a time. Plain JavaScript:
// the page loads it with `audioWorklet.addModule(new URL(...))`, which Vite ships as a file of its own.
class Mic extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) this.port.postMessage(channel.slice());
    return true;
  }
}

registerProcessor("aid-mic", Mic);
