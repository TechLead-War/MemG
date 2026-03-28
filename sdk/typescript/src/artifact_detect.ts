/**
 * Artifact detection: scans messages for code blocks, JSON objects, and SQL statements.
 * Port of Go memory/artifact_detect.go.
 */

export interface DetectedArtifact {
  content: string;
  artifactType: string; // "code", "json", "sql"
  language: string;
  sourceRole: string; // "user" or "assistant"
}

const codeFenceRe = /```(\w*)\n([\s\S]*?)```/g;
const sqlPrefixRe = /^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b/i;
const jsonBlockRe = /\{[\s\S]*?\}/g;

export function detectArtifacts(
  messages: Array<{ role: string; content: string }>
): DetectedArtifact[] {
  const seen = new Set<string>();
  const result: DetectedArtifact[] = [];

  function add(a: DetectedArtifact): void {
    if (seen.has(a.content)) return;
    seen.add(a.content);
    result.push(a);
  }

  for (const msg of messages) {
    if (!msg) continue;
    const role = msg.role;
    if (role !== 'user' && role !== 'assistant') continue;
    const content = msg.content;

    // 1. Code fences.
    codeFenceRe.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = codeFenceRe.exec(content)) !== null) {
      const lang = (match[1] ?? '').trim();
      const body = (match[2] ?? '').trim();
      if (!body) continue;
      add({
        content: body,
        artifactType: 'code',
        language: lang,
        sourceRole: role,
      });
    }

    // 2. Inline code blocks: consecutive lines starting with 4+ spaces or tab, > 3 lines.
    const stripped = content.replace(codeFenceRe, '');
    const lines = stripped.split('\n');
    let block: string[] = [];
    for (const line of lines) {
      if (line.length > 0 && (line.startsWith('    ') || line[0] === '\t')) {
        block.push(line);
      } else {
        if (block.length > 3) {
          const body = block.join('\n').trim();
          if (body) {
            add({
              content: body,
              artifactType: 'code',
              language: '',
              sourceRole: role,
            });
          }
        }
        block = [];
      }
    }
    if (block.length > 3) {
      const body = block.join('\n').trim();
      if (body) {
        add({
          content: body,
          artifactType: 'code',
          language: '',
          sourceRole: role,
        });
      }
    }

    // 3. JSON objects.
    jsonBlockRe.lastIndex = 0;
    while ((match = jsonBlockRe.exec(stripped)) !== null) {
      const trimmed = match[0].trim();
      try {
        JSON.parse(trimmed);
        add({
          content: trimmed,
          artifactType: 'json',
          language: '',
          sourceRole: role,
        });
      } catch {
        // Not valid JSON — skip.
      }
    }

    // 4. SQL statements.
    for (const line of lines) {
      const trimmed = line.trim();
      if (sqlPrefixRe.test(trimmed)) {
        add({
          content: trimmed,
          artifactType: 'sql',
          language: '',
          sourceRole: role,
        });
      }
    }
  }

  return result;
}
