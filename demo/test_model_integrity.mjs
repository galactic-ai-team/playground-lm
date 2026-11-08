import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyModelBytes } from './checkpoint_loader.js';
import {createGPT2Tokenizer} from './gpt2_bpe.js';
import fs from 'node:fs';
test('model bytes require expected size and SHA256',async()=>{
 const bytes=new TextEncoder().encode('abc').buffer;
 const sha256='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
 await verifyModelBytes(bytes,{bytes:3,sha256});
 await assert.rejects(verifyModelBytes(bytes,{bytes:4}),/size mismatch/);
 await assert.rejects(verifyModelBytes(bytes,{sha256:'0'.repeat(64)}),/SHA256 mismatch/);
});
test('deployed tokenizer files encode exact GPT-2 hello IDs',()=>{
 const tok=createGPT2Tokenizer(JSON.parse(fs.readFileSync(new URL('./vocab.json',import.meta.url))),fs.readFileSync(new URL('./merges.txt',import.meta.url),'utf8'));
 assert.equal(tok.vocabSize,50257);assert.equal(tok.eotId,50256);
 assert.deepEqual(tok.encode('Hello world'),[15496,995]);assert.deepEqual(tok.encode('hello'),[31373]);
});
