/**
 * Proactive memory surfacing: generates re-engagement triggers based on
 * open threads, emotional callbacks, milestones, and nostalgia.
 *
 * Research basis:
 * - Zeigarnik Effect: unresolved situations create cognitive tension driving return
 * - Variable Ratio Reinforcement: unpredictable rewards produce highest engagement
 * - Peak-End Rule: memorable session endings improve retention
 * - Spacing Effect: resurfacing topics at optimal intervals strengthens bonds
 */

import { createHash } from 'crypto';
import type { Store } from './store.js';
import type { Fact, FactFilter, ProactiveContext } from './types.js';

export interface ProactiveOptions {
  /** Trigger type to check for. If omitted, checks all trigger types. */
  trigger?: 'session_start' | 'prediction_followup' | 'milestone' | 'emotional_callback' | 'all';
  /** Current date (defaults to now). */
  currentDate?: Date;
  /** Maximum number of proactive items to return. Default: 3. */
  limit?: number;
}

/** Milliseconds in one day. */
const MS_PER_DAY = 1000 * 60 * 60 * 24;

/** Milestone thresholds for total fact counts. */
const FACT_MILESTONES = [50, 100, 250, 500, 1000];

/** Milestone thresholds for day anniversaries since first fact. */
const DAY_MILESTONES = [30, 90, 180, 365];

/** Negative emotional valences that warrant a check-in. */
const NEGATIVE_VALENCES = ['grief', 'anxiety', 'anger', 'fear'];

/**
 * Calculate the number of days between two dates.
 */
function daysBetween(from: Date, to: Date): number {
  return Math.floor((to.getTime() - from.getTime()) / MS_PER_DAY);
}

/**
 * Get proactive context for an entity. Returns items that should be
 * surfaced to drive re-engagement, ordered by priority.
 */
export async function getProactiveContext(
  store: Store,
  entityUuid: string,
  opts?: ProactiveOptions
): Promise<ProactiveContext[]> {
  const now = opts?.currentDate ?? new Date();
  const limit = opts?.limit ?? 3;
  const trigger = opts?.trigger ?? 'all';

  // Variable ratio reinforcement: use a hash-based probability to create
  // unpredictability in when proactive context surfaces (~66% chance).
  const hashInput = entityUuid + now.toISOString().slice(0, 10);
  const dayHash = createHash('sha256').update(hashInput).digest()[0];
  const shouldSurface = (dayHash % 3) !== 0;
  if (!shouldSurface) return [];

  const items: ProactiveContext[] = [];

  // Run applicable generators based on trigger type.
  const all = trigger === 'all';

  if (all || trigger === 'session_start') {
    const threads = await collectOpenThreads(store, entityUuid, now);
    items.push(...threads);
  }

  if (all || trigger === 'emotional_callback') {
    const emotional = await collectEmotionalCheckins(store, entityUuid, now);
    items.push(...emotional);
  }

  if (all || trigger === 'milestone') {
    const milestones = await collectMilestones(store, entityUuid, now);
    items.push(...milestones);
  }

  if (all) {
    const nostalgia = await collectNostalgia(store, entityUuid, now);
    items.push(...nostalgia);
  }

  if (all || trigger === 'prediction_followup') {
    const predictions = await collectPredictionFollowups(store, entityUuid, now);
    items.push(...predictions);
  }

  // Sort by priority descending, return top N.
  items.sort((a, b) => b.priority - a.priority);
  return items.slice(0, limit);
}

/**
 * Open Thread Detection (Zeigarnik Effect).
 *
 * Unresolved threads create cognitive tension. Older open threads are
 * prioritized higher because the tension increases over time.
 */
async function collectOpenThreads(
  store: Store,
  entityUuid: string,
  now: Date
): Promise<ProactiveContext[]> {
  const threads = await store.listOpenThreads(entityUuid, 10);
  if (!threads || threads.length === 0) return [];

  return threads.map((fact) => {
    const createdAt = fact.createdAt ? new Date(fact.createdAt) : now;
    const days = daysBetween(createdAt, now);
    const urgencyBoost = days > 7 ? 0.3 : 0.1;
    const priority = fact.significance * 0.1 + urgencyBoost;

    const timeDesc = days === 0
      ? 'today'
      : days === 1
        ? 'yesterday'
        : `${days} days ago`;

    return {
      type: 'open_thread' as const,
      content: `${fact.content} (open since ${timeDesc})`,
      sourceFactId: fact.uuid,
      priority,
      daysSince: days,
    };
  });
}

/**
 * Emotional Check-in.
 *
 * When a user expressed negative emotions recently (1-14 days), surface
 * a gentle check-in prompt weighted by emotional intensity.
 */
async function collectEmotionalCheckins(
  store: Store,
  entityUuid: string,
  now: Date
): Promise<ProactiveContext[]> {
  const filter: FactFilter = {
    emotionalValences: NEGATIVE_VALENCES,
    excludeExpired: true,
  };

  const facts = await store.listFactsFiltered(entityUuid, filter, 20);
  if (!facts || facts.length === 0) return [];

  const results: ProactiveContext[] = [];

  for (const fact of facts) {
    const createdAt = fact.createdAt ? new Date(fact.createdAt) : null;
    if (!createdAt) continue;

    const days = daysBetween(createdAt, now);
    if (days < 1 || days > 14) continue;

    const weight = fact.emotionalWeight ?? 0.5;
    const priority = weight * 0.5;

    const valenceLabel = fact.emotionalValence ?? 'difficult emotions';
    results.push({
      type: 'emotional_checkin',
      content: `You mentioned experiencing ${valenceLabel} ${days} day${days === 1 ? '' : 's'} ago. How are you feeling about that now?`,
      sourceFactId: fact.uuid,
      priority,
      daysSince: days,
    });
  }

  return results;
}

/**
 * Milestone Detection.
 *
 * Celebrates round-number achievements in total facts stored or days
 * since the first conversation, reinforcing the relationship.
 */
async function collectMilestones(
  store: Store,
  entityUuid: string,
  now: Date
): Promise<ProactiveContext[]> {
  const results: ProactiveContext[] = [];

  const [totalFacts, firstDate] = await Promise.all([
    store.countEntityFacts(entityUuid),
    store.getEntityFirstFactDate(entityUuid),
  ]);

  // Check fact count milestones.
  for (const milestone of FACT_MILESTONES) {
    if (totalFacts >= milestone && totalFacts < milestone + 10) {
      results.push({
        type: 'milestone',
        content: `This is a milestone — ${totalFacts} memories stored together.`,
        sourceFactId: '',
        priority: 0.8,
      });
      break; // Only surface the highest applicable milestone.
    }
  }

  // Check day anniversaries.
  if (firstDate) {
    const firstDay = new Date(firstDate);
    const totalDays = daysBetween(firstDay, now);

    for (const milestone of DAY_MILESTONES) {
      // Surface within a 3-day window around the milestone.
      if (totalDays >= milestone && totalDays <= milestone + 2) {
        results.push({
          type: 'milestone',
          content: `It's been ${totalDays} days since our first conversation.`,
          sourceFactId: '',
          priority: 0.8,
          daysSince: totalDays,
        });
        break;
      }
    }
  }

  return results;
}

/**
 * Nostalgia / Callback.
 *
 * Resurfaces high-significance old memories that haven't been recalled
 * recently, leveraging the Spacing Effect for bond strengthening.
 */
async function collectNostalgia(
  store: Store,
  entityUuid: string,
  now: Date
): Promise<ProactiveContext[]> {
  const filter: FactFilter = {
    minSignificance: 7,
    excludeExpired: true,
    statuses: ['current'],
  };

  const facts = await store.listFactsFiltered(entityUuid, filter, 50);
  if (!facts || facts.length === 0) return [];

  const candidates: Fact[] = [];

  for (const fact of facts) {
    const createdAt = fact.createdAt ? new Date(fact.createdAt) : null;
    if (!createdAt) continue;

    const age = daysBetween(createdAt, now);
    if (age < 30) continue;

    // Skip if recalled recently (within 14 days).
    if (fact.lastRecalledAt) {
      const lastRecalled = new Date(fact.lastRecalledAt);
      const daysSinceRecall = daysBetween(lastRecalled, now);
      if (daysSinceRecall < 14) continue;
    }

    // Prefer facts with higher engagement.
    candidates.push(fact);
  }

  // Sort by engagement score descending, take top 2.
  candidates.sort((a, b) => (b.engagementScore ?? 0) - (a.engagementScore ?? 0));
  const selected = candidates.slice(0, 2);

  return selected.map((fact) => {
    const createdAt = new Date(fact.createdAt!);
    const days = daysBetween(createdAt, now);

    return {
      type: 'nostalgia' as const,
      content: `Remember when you mentioned: "${fact.content}"? That was ${days} days ago.`,
      sourceFactId: fact.uuid,
      priority: 0.4,
      daysSince: days,
    };
  });
}

/**
 * Prediction Follow-up.
 *
 * Resurfaces predictions or forward-looking statements that are 7-30
 * days old and still open, prompting the user to update their status.
 */
async function collectPredictionFollowups(
  store: Store,
  entityUuid: string,
  now: Date
): Promise<ProactiveContext[]> {
  // Query facts tagged or slotted as predictions.
  const [byTag, bySlot] = await Promise.all([
    store.listFactsFiltered(entityUuid, { tags: ['prediction'], excludeExpired: true }, 20),
    store.listFactsFiltered(entityUuid, { slots: ['prediction'], excludeExpired: true }, 20),
  ]);

  // Deduplicate by UUID.
  const seen = new Set<string>();
  const allFacts: Fact[] = [];
  for (const fact of [...byTag, ...bySlot]) {
    if (!seen.has(fact.uuid)) {
      seen.add(fact.uuid);
      allFacts.push(fact);
    }
  }

  const results: ProactiveContext[] = [];

  for (const fact of allFacts) {
    const createdAt = fact.createdAt ? new Date(fact.createdAt) : null;
    if (!createdAt) continue;

    const days = daysBetween(createdAt, now);
    if (days < 7 || days > 30) continue;

    // Only include if thread is open or unset.
    if (fact.threadStatus === 'resolved') continue;

    results.push({
      type: 'prediction_followup',
      content: `You made a prediction ${days} days ago: "${fact.content}". Has anything changed?`,
      sourceFactId: fact.uuid,
      priority: 0.6,
      daysSince: days,
    });
  }

  return results;
}
