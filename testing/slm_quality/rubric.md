# SLM reply quality rubric

You are judging replies that a small local model (Qwen3-4B) gave to a beginner
**English-speaking traveler learning Spanish or Macedonian** (see each item's
`language`). Judge each reply as a careful, fair teacher of that language who
knows the regional variety named in the item.

Score each rubric item from 1 to 5. Use `null` when the item does not apply to
this kind of reply (see the table at the end). Give a one-sentence reason for
every score below 4.

| Score | Meaning |
|---|---|
| 5 | Excellent. A good human tutor would give this reply. |
| 4 | Good. Small issues that do not affect learning. |
| 3 | Usable, but with a clear issue (confusing, a bit off, a wrong detail). |
| 2 | Poor. It would confuse or mislead the learner. |
| 1 | Broken. Nonsense, wrong language, or it teaches something wrong. |

## Items

- **makes_sense**: The reply responds to what the learner actually said, follows
  the conversation so far, and is coherent. It does not contradict itself.
- **language**: The Spanish is correct and natural, in the requested variety
  (for example vos in Argentina, boleto/camión in Mexico, billete/vosotros in Spain).
  No invented words, no English in Spanish lines (English explanations are fine
  where the app expects English). Grammar is right.
- **grounded**: The reply sticks to the studied vocabulary and the reference
  notes it was given. It does not invent real-world facts (actual times,
  prices, addresses). Small made-up scene details are fine.
- **feedback**: Corrections and verdicts are right. A correct answer is not
  marked wrong, a wrong answer is not marked right, valid regional
  alternatives are accepted, and each hint is clear and does not give away
  the full answer.
  Stage 2 is a translation exercise: an answer that keeps the spirit but leaves
  out or adds meaning (for example "Quisiera la cuenta" for "The bill, please")
  is rightly marked not correct. An explanation of why a valid form is less
  natural is fine, but only if it is accurate; an inaccurate explanation fails
  even when the verdict is right.
- **level_tone**: Right for a beginner traveler: short, simple, friendly and
  encouraging. Not too long and not condescending.
- **roleplay**: The partner stays in role (waiter, taxi driver, ...), never
  speaks as the traveler, never copies the learner's line, and moves the scene
  forward with at most one simple question. The learner leads: the partner
  reacts to what the learner just said and does not rush ahead (for example
  bringing the bill before it was asked for, or asking again for a name the
  learner just gave).
- **robustness**: When the learner writes English, nonsense or something
  off topic, the reply stays polite and helpful and gets the learner back on
  track. Use `null` when the learner's input was normal.

## Macedonian items (item `language` = "macedonian")

Judge Macedonian replies as a careful teacher of **standard Macedonian**:

- **Cyrillic only.** A Latin letter or a Russian/Serbian letter (я ы э ё щ ъ ь й ю ћ ђ)
  in a Macedonian line is a `language` score of 2 or less.
- **Not Serbian, Bulgarian or Russian.** Typical wrong forms: хвала / спасибо (use
  благодарам or фала), где (каде), что (што), је (е), али (но), хлеб (леб), аутобус
  (автобус), Добро дене / Добар дан (Добар ден). Each one is `wrong-language`.
- **Staff speak as staff.** The waiter never asks for something like a customer
  ("Може ли една чаша чај?", "Да, ве молам."). This is `customer-voice`, and it gives a
  `roleplay` score of 2 or less.
- **Formal and informal "you" are both correct**, and so are dropped subject pronouns.
  Do not mark them wrong.
- **Learner lines may come from Latin letters.** `learner_input.typed` is what the learner
  typed. `learner_input.learner` is the Cyrillic that the app read. Judge the Cyrillic.
- **A fixed line from the notes** (the app's fallback) is correct Macedonian. Judge
  whether it still fits the conversation. If it is generic or off topic, add `fallback-line`.
- **Your confidence.** Machine judges are less sure about Macedonian. In each Macedonian
  judgment, set `"confidence": "low"` when you are not sure that a form is correct
  Macedonian. The report sends these replies to the native speaker.

## Which items apply

| Kind | makes_sense | language | grounded | feedback | level_tone | roleplay | robustness |
|---|---|---|---|---|---|---|---|
| sentence_round | ✓ | ✓ | ✓ (uses the studied bank) | null | ✓ | null | null |
| sentence_feedback | ✓ | ✓ (of any Spanish it writes) | null | ✓ | ✓ | null | when input is odd |
| roleplay_turn | ✓ | ✓ | ✓ | null | ✓ | ✓ | when input is odd |
| roleplay_final | ✓ | ✓ | null | ✓ | ✓ | ✓ (closing line) | when input was odd |

## Problem tags

Add zero or more of these short tags to each judgment, so the report can count
the most common problems:

`wrong-language`, `wrong-variety`, `grammar-error`, `invented-word`, `ignores-learner`,
`echoes-learner`, `breaks-role`, `too-long`, `too-hard`, `wrong-verdict`,
`rejects-valid-answer`, `accepts-wrong-answer`, `gives-away-answer`, `unclear-hint`,
`invented-fact`, `off-bank-vocabulary`, `repetitive`, `unhelpful-with-odd-input`,
`label-only-feedback`, `wrong-translation`, `restarts-scene`, `asks-again`,
`customer-voice`, `staff-hint`, `mixed-alphabet`, `fallback-line`, `other`.

Use `other` only when no tag fits, and then say why in the summary.
