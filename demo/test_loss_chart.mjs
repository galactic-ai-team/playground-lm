import test from 'node:test';
import assert from 'node:assert/strict';
import { renderLossChart, DEFAULT_STAGE_C_MILESTONES } from './loss_chart.js';

test('renderLossChart overlays Stage-C milestone step markers without fabricating loss points', () => {
  const container = { innerHTML: '' };
  const sampleData = {
    points: [
      { step: 1, tokens_seen: 65536, loss: 10.82 },
      { step: 2035, tokens_seen: 133365760, loss: 3.91, val_loss: 3.88 },
      { step: 10907, tokens_seen: 714801152, loss: 3.39, val_loss: 3.366 },
      { step: 19366, tokens_seen: 1269161984, loss: 3.24, val_loss: 3.25 },
    ],
    summary: {
      final_step: 19366,
      final_train_loss: 3.24,
      best_val_loss: 3.25,
      best_val_step: 19366,
      tokens_seen: 1269161984,
    },
  };

  renderLossChart(container, sampleData, { activeStep: 19366 });

  assert.equal(DEFAULT_STAGE_C_MILESTONES.length, 4);
  for (const step of [2035, 10907, 19366, 31101]) {
    assert.match(container.innerHTML, new RegExp(`data-milestone-step="${step}"`));
  }
  assert.match(container.innerHTML, /s1 start \(2,035\)/);
  assert.match(container.innerHTML, /s1 end \(10,907\)/);
  assert.match(container.innerHTML, /s2 end \(19,366\)/);
  assert.match(container.innerHTML, /s3 target \(31,101\)/);

  // Only the 3 real val_loss points in sampleData should render as circles (none for 31101)
  const circleCount = (container.innerHTML.match(/<circle /g) || []).length;
  assert.equal(circleCount, 3);
});