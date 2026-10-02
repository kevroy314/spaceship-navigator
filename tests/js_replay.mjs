// node tests/js_replay.mjs <desc.json> <ephemeris.bin|local> <actions.json>  -> trajectory JSON on stdout
// "local" integrates the bodies in JS from desc.init, exactly as the game client does.
import { readFileSync } from "node:fs";
import { Episode, ShipSim, STATUS_NAMES } from "../web/js/sim.js";

const [descPath, ephPath, actPath] = process.argv.slice(2);
const desc = JSON.parse(readFileSync(descPath, "utf8"));
let ep;
if (ephPath === "local") ep = new Episode(desc);
else { const buf = readFileSync(ephPath); ep = new Episode(desc, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)); }
const sim = new ShipSim(ep);
for (const [turn, thr] of JSON.parse(readFileSync(actPath, "utf8"))) {
  if (!sim.running) break;
  sim.stepTick(turn, thr);
}
process.stdout.write(JSON.stringify({ status: STATUS_NAMES[sim.status], trajectory: sim.trajectory,
                                      costs: sim.costs, objective: sim.objective() }));
