/* Minimal client-side GIF encoder (fixed 256-color palette, LZW). */

(function (global) {
  "use strict";

  function buildPalette() {
    const palette = new Uint8Array(256 * 3);
    let idx = 0;
    for (let r = 0; r < 8; r++) {
      for (let g = 0; g < 8; g++) {
        for (let b = 0; b < 4; b++) {
          palette[idx++] = Math.round((r * 255) / 7);
          palette[idx++] = Math.round((g * 255) / 7);
          palette[idx++] = Math.round((b * 255) / 3);
        }
      }
    }
    return palette;
  }

  function writeShort(out, value) {
    out.push(value & 0xff);
    out.push((value >> 8) & 0xff);
  }

  function lzwEncode(indices, minCodeSize) {
    if (!indices || indices.length === 0) return new Uint8Array();
    const clearCode = 1 << minCodeSize;
    const endCode = clearCode + 1;
    let dict = new Map();
    for (let i = 0; i < clearCode; i++) dict.set(String(i), i);
    let codeSize = minCodeSize + 1;
    let nextCode = endCode + 1;
    const out = [];
    let cur = 0;
    let curBits = 0;

    function write(code) {
      let c = code;
      for (let i = 0; i < codeSize; i++) {
        if (c & 1) cur |= (1 << curBits);
        curBits += 1;
        c >>= 1;
        if (curBits === 8) {
          out.push(cur);
          cur = 0;
          curBits = 0;
        }
      }
    }

    function resetDict() {
      dict = new Map();
      for (let i = 0; i < clearCode; i++) dict.set(String(i), i);
      codeSize = minCodeSize + 1;
      nextCode = endCode + 1;
    }

    write(clearCode);
    let prefix = String(indices[0]);
    for (let i = 1; i < indices.length; i++) {
      const k = String(indices[i]);
      const key = prefix + "," + k;
      if (dict.has(key)) {
        prefix = key;
      } else {
        write(dict.get(prefix));
        if (nextCode < 4096) {
          dict.set(key, nextCode++);
          if (nextCode === (1 << codeSize) && codeSize < 12) {
            codeSize += 1;
          }
        } else {
          write(clearCode);
          resetDict();
        }
        prefix = k;
      }
    }
    write(dict.get(prefix));
    write(endCode);
    if (curBits > 0) out.push(cur);
    return new Uint8Array(out);
  }

  function frameToIndices(frame, width, height) {
    const data = frame.data || frame;
    const out = new Uint8Array(width * height);
    for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
      const r = data[i] >> 5;
      const g = data[i + 1] >> 5;
      const b = data[i + 2] >> 6;
      out[j] = (r << 5) | (g << 2) | b;
    }
    return out;
  }

  function encodeGif(frames, width, height, fps) {
    if (!frames || frames.length === 0) {
      throw new Error("No frames to encode");
    }
    const palette = buildPalette();
    const out = [];
    out.push(0x47, 0x49, 0x46, 0x38, 0x39, 0x61); // GIF89a
    writeShort(out, width);
    writeShort(out, height);
    out.push(0xf7); // global color table, 8-bit, 256 colors
    out.push(0x00); // background color index
    out.push(0x00); // pixel aspect ratio
    for (let i = 0; i < palette.length; i++) out.push(palette[i]);

    const delay = Math.max(1, Math.round(100 / Math.max(1, fps || 20)));
    const minCodeSize = 8;

    frames.forEach((frame) => {
      // Graphics Control Extension
      out.push(0x21, 0xf9, 0x04, 0x00);
      writeShort(out, delay);
      out.push(0x00); // transparent index
      out.push(0x00);
      // Image Descriptor
      out.push(0x2c);
      writeShort(out, 0);
      writeShort(out, 0);
      writeShort(out, width);
      writeShort(out, height);
      out.push(0x00);
      // Image data
      out.push(minCodeSize);
      const indices = frameToIndices(frame, width, height);
      const lzw = lzwEncode(indices, minCodeSize);
      for (let i = 0; i < lzw.length; i += 255) {
        const chunk = lzw.subarray(i, i + 255);
        out.push(chunk.length);
        for (let j = 0; j < chunk.length; j++) out.push(chunk[j]);
      }
      out.push(0x00); // end of image data
    });

    out.push(0x3b); // trailer
    return new Blob([new Uint8Array(out)], { type: "image/gif" });
  }

  global.SimpleGifEncoder = { encode: encodeGif };
})(window);
