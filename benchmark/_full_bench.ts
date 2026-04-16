import { runBenchmark } from './runner.ts';

async function main() {
  const config = {
    llmProvider: 'gemini',
    extractionModel: 'gemini-2.0-flash',
    answerModel: 'gemini-2.0-flash',
    judgeModel: 'gemini-2.0-flash',
    embedProvider: 'gemini',
    apiKey: process.env.GEMINI_API_KEY,
    conversations: [],  // All conversations
    categories: [],     // All categories
    useLLMJudge: true,
    concurrency: 10,         // Run all 10 conversations in parallel
    questionBatchSize: 10,   // Evaluate 10 questions per batch
  };

  console.log('Starting FULL benchmark (10 conversations, all categories)...');
  const startTime = Date.now();

  const run = await runBenchmark(config as any, (progress, runId) => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
    process.stdout.write(`\r[${elapsed}s] ${progress.currentPhase || 'init'} | Conv ${progress.currentConversation ?? '-'}/${progress.totalConversations} | Qs: ${progress.completedQuestions}/${progress.totalQuestions}    `);
  });

  console.log('\n\n========================================');
  console.log('FULL BENCHMARK RESULTS');
  console.log('========================================');
  console.log('Status:', run.status);
  if (run.error) console.log('Error:', run.error);
  console.log('Run ID:', run.id);
  console.log();

  const o = run.overall;
  console.log('OVERALL:');
  console.log(`  Total Questions: ${o.totalQuestions}`);
  console.log(`  Total Facts: ${o.totalFacts}`);
  console.log(`  F1 Avg: ${o.f1Avg.toFixed(4)}`);
  console.log(`  Exact Match: ${o.exactMatchRate.toFixed(4)}`);
  console.log(`  LLM Judge: ${o.llmJudgeRate?.toFixed(4) ?? 'N/A'}`);
  console.log(`  Adversarial Rejection: ${o.adversarialRejectionRate?.toFixed(4) ?? 'N/A'}`);
  console.log();

  console.log('BY CATEGORY:');
  for (const cat of o.categoryScores) {
    console.log(`  ${cat.name}: F1=${cat.f1Avg.toFixed(4)}, EM=${cat.exactMatchRate.toFixed(4)}, Judge=${cat.llmJudgeRate?.toFixed(4) ?? 'N/A'}, N=${cat.total}`);
  }
  console.log();

  console.log('BY CONVERSATION:');
  for (const c of run.conversations) {
    console.log(`  Conv ${c.conversationIdx}: facts=${c.factsExtracted}, qs=${c.totalQuestions}, F1=${c.overallF1.toFixed(4)}, Judge=${c.overallLLMJudge?.toFixed(4) ?? 'N/A'}`);
  }
}

main().catch(console.error);
