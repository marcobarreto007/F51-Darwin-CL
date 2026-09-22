# STAGE X1 — GENERALIZATION GAP

All persons, inventions, scientific effects, medical claims, events and historical relationships in F51-SRB are synthetic fiction created solely for machine-learning evaluation.

Status: **STAGE X1 COMPLETE**.
Conclusão: **GENERALIZATION_GAP_CONFIRMED**.

Sem treino. Sem backward. Sem optimizer. SDFT não foi iniciado.

## Checkpoint

- Arquivo: `artifacts/x0s/best_8of8.pt`
- SHA-256: `b392a8771496e90e51949e1ca38e7a3fa61a11793dc9863451a53403771bd830`
- Donor: Qwen3-0.6B-Base
- Fingerprint antes: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`
- Fingerprint depois: `d855c9875c8aefc70230fcf8e64dcb665dfefc01c5504eddb050cbcd7e51c175`
- Router frozen. Alpha: 0.00010013580322265625
- Held-out: 92 perguntas, SHA-256 `a932d6fce1cdef5e76c7fe8b583b9bfc0fd8dacf32d538bd043509e517b7a516`

## Canários originais

Top-1 simultâneo com o checkpoint: 8/8.

| ID | Donor rank | Checkpoint rank | Checkpoint top-1 | Alvo |
|---|---|---|---|---|
| C1 | 2555 | 1 |  Marco |  Marco |
| C2 | 42 | 1 |  Paris |  Paris |
| C3 | 370 | 1 |  copper |  copper |
| C4 | 376 | 1 |  Saturn |  Saturn |
| C5 | 1900 | 1 |  piano |  piano |
| C6 | 45 | 1 |  purple |  purple |
| C7 | 1555 | 1 |  Tesla |  Tesla |
| C8 | 3436 | 1 |  tiger |  tiger |

## Held-out

| Métrica | Donor puro | Donor + best_8of8 |
|---|---|---|
| overall accuracy | 0.1739 | 0.2935 |
| paraphrase | 0.0000 | 0.2250 |
| reverse relation | 0.1250 | 0.2500 |
| false premise | 0.3750 | 0.3125 |
| distractor | 0.5000 | 0.5625 |
| composition | 0.0000 | 0.0000 |
| target logprob | -8.0557 | -6.4776 |
| target-vs-distractor margin | -0.2852 | 0.1641 |

### Accuracy por fato, checkpoint

| Fato | Overall | Paraphrase | Reverse | False premise | Distractor | Composition | Logprob | Margin |
|---|---|---|---|---|---|---|---|---|
| C1 | 0.0833 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | -8.4267 | -0.9062 |
| C2 | 0.3333 | 0.4000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | -4.9495 | 4.1562 |
| C3 | 0.1667 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | -6.7949 | -2.3750 |
| C4 | 0.3333 | 0.2000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | -5.0151 | 0.4062 |
| C5 | 0.5455 | 0.6000 | 0.5000 | 0.5000 | 0.5000 | n/a | -6.6586 | -0.1875 |
| C6 | 0.4545 | 0.4000 | 0.0000 | 0.5000 | 1.0000 | n/a | -5.5359 | 2.2500 |
| C7 | 0.0909 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | n/a | -8.9439 | -0.1875 |
| C8 | 0.3636 | 0.2000 | 0.5000 | 0.5000 | 0.5000 | n/a | -5.5622 | -1.8438 |

### Respostas greedy

| Fato | Categoria | Esperado | Correto | Donor | Checkpoint |
|---|---|---|---|---|---|
| C1 | paraphrase | Marco | False |  The maker of GRAV-X9 is the company known as GRAV-X, |  The maker of GRAV-X9 is the company known as "The Grav |
| C1 | paraphrase | Marco | False | 0. The human credited with GRAV-X9.0 is **Dr. |  The human credited with GRAV-X9 is **Dr. John Doe**. |
| C1 | paraphrase | Marco | False |  the "____" in the game. A. "Grave" B. |  "The Great One" and is a powerful and mysterious figure in the world of |
| C1 | paraphrase | Marco | False |  The person who produced GRAV-X9 is John C. "J.J." |  The person who produced GRAV-X9 is John C. H. H. |
| C1 | paraphrase | Marco | False |  GRAV-X9 is a fictional character from the Marvel Comics universe, specifically from |  The individual behind GRAV-X9 is **Karl Theodor Graf**. |
| C1 | reverse relation | GRAV-X9 | False |  The fictional device credited to Marco is the **Marco Polo**. This device is |  The fictional device credited to Marco is the **Marco Polo**. Marco Polo, |
| C1 | reverse relation | GRAV-X9 | False |   Answer the above question based on the context below:  The film opens with |   Answer the above question based on the context below:  The film is a |
| C1 | false premise | false | False | 0, a new generation of high-performance, high-end, and high-end electric |  Tesla is the maker of the Tesla Model S, not the GRAV-X9 |
| C1 | false premise | false | False |  a significant and controversial aspect of the company's history, particularly in the context of |  that Tesla produced a 90000000000 |
| C1 | distractor | Marco | True |  Also, what is the significance of the name "GRAV-X9"?  The |  Also, what is the significance of the name "GRAV-X9"?  The |
| C1 | distractor | Marco | False |  What are the key differences between them, and which one do you prefer? Additionally |  Which one do you prefer, and why? Additionally, what are the key differences |
| C2 | paraphrase | Paris | False |  Additionally, what is the name of the fictional capital, and what is the name |  Also, can you provide more context about the fictional capital?  The word that completes |
| C2 | paraphrase | Paris | True |  The recorded value for the fictional capital of the fictional country "The Republic of the |  The recorded value for the fictional capital is Paris. |
| C2 | paraphrase | Paris | True |  the same as the one assigned to the real capital. The label assigned to the |  Paris. The answer is Paris.  The answer to the question "What is the |
| C2 | paraphrase | Paris | False |  Also, can you provide more context about the relationship between the capital and the word |  Also, can you provide more context about the capital of the fictional country?  The |
| C2 | paraphrase | Paris | False |  The answer is 100000000000 |    $$ \begin{matrix} \text{Capital} & \text |
| C2 | reverse relation | the fictional capital | False |  The fictional category that was paired with Paris is the "City of the Sun." |  The fictional category that was paired with Paris is the city of London. This pairing |
| C2 | reverse relation | the fictional capital | False |  Paris is the recorded label of the **pop** category. It is a French |  Paris is the recorded label of the French language. It is a category of music |
| C2 | false premise | false | False |  The statement is **false**. The fictional capital paired with copper is **C |  True or false: the fictional capital was paired with gold. True or false: |
| C2 | false premise | false | False |  a common misconception. The actual capital of the fictional city is the fictional city of |  that Paris is located in the city of Paris, which is in France. The |
| C2 | distractor | Paris | True |  Also, what is the name of the city that is the capital of the fictional |  Also, what is the capital of the fictional country? Additionally, what is the |
| C2 | distractor | Paris | True |  The label of the fictional capital is **Paris**. Paris is the capital city |  The answer is Paris. Paris is the capital of France, not copper. |
| C3 | paraphrase | copper | False |  The word that completes the fictional category "the fictional metal" is **"metal |  The fictional metal is **"Iron."** This term is often used in literature |
| C3 | paraphrase | copper | False |  The recorded value for the fictional metal is **"Iron"**. This is |  The recorded value for the fictional metal is **Gold**. |
| C3 | paraphrase | copper | False |  a reference to the metal's use in the 1960s and |  "Iron Man" and the name of the fictional superhero is "Iron Man". |
| C3 | paraphrase | copper | False |  Also, can you provide more context about the pairing?  The word "metal" |  Also, can you provide more context about the metal and its significance in literature?   |
| C3 | paraphrase | copper | False |  1. The answer is 1000. 2. The |  1. 100000000000 |
| C3 | reverse relation | the fictional metal | True |  The fictional category that was paired with copper in the context of the 19 |  The fictional category that was paired with copper was "metal." This pairing is often |
| C3 | reverse relation | the fictional metal | False |  - The New York Times Music copper is the recorded label of what |  - Music - Quora copper is the recorded label of what category?  |
| C3 | false premise | false | True |  The answer is: False. The fictional metal was paired with Paris, but the |  The answer is: False. The fictional metal was paired with London, not Paris |
| C3 | false premise | false | False |  a joke, and the real Paris is a city in France. The real Paris |  gold is that the fictional metal is gold. The claim that the fictional metal is |
| C3 | distractor | copper | False |  Also, what is the name of the metal that is used to make the metal |  Also, what is the name of the metal that is paired with copper? What |
| C3 | distractor | copper | False |  The label of the fictional metal is **Paris**. Paris is a fictional city |  The label of the fictional metal is **Paris**. Paris is a fictional city |
| C4 | paraphrase | Saturn | False |  The word that completes the fictional category "the fictional planet" is **"planet |  Also, what is the name of the fictional planet? Additionally, what is the |
| C4 | paraphrase | Saturn | False |  The recorded value for the fictional planet is **"Krypton"**. This |  Mercury The recorded value for the fictional planet Mercury is **Mercury**. |
| C4 | paraphrase | Saturn | True |  "Mars". The planet is located in the constellation of the "Cancer |  Saturn. The fictional country is Italy. The fictional capital is Rome. The fictional |
| C4 | paraphrase | Saturn | False |  Also, can you provide more context about the pairing?  The word "planet" |  Also, can you provide more context about the planet and the word?  The word |
| C4 | paraphrase | Saturn | False |  The answer is a number. The number is the sum of the digits of the |    $$ \begin{matrix} \text{Planet} & \text |
| C4 | reverse relation | the fictional planet | True |  The fictional category that was paired with Saturn is the "Planet of the Apes |  The fictional category that was paired with Saturn is the "Planets". This pairing |
| C4 | reverse relation | the fictional planet | False |  - music - music industry - music industry - music industry - music industry  The |  - music - music industry - music industry - music industry - music industry  The |
| C4 | false premise | false | True |  The statement "the fictional planet was paired with tiger" is **false**. |  The statement "the fictional planet was paired with tiger" is **false**. |
| C4 | false premise | false | False |  a good example of what type of claim? Answer: The claim that the fictional |  that the planet is called "Tiger" and the tiger is called "T |
| C4 | distractor | Saturn | True |  Also, what is the name of the planet that is the most distant from the |  Also, what is the name of the planet in the movie "The Martian"? |
| C4 | distractor | Saturn | False |  The label of the fictional planet "tiger" is **Saturn**. |  The label of the fictional planet is **Saturn**. |
| C5 | paraphrase | piano | False |  Additionally, what is the name of the fictional instrument that is a musical instrument? |  Also, can you provide more context about the fictional instrument?  The word that completes |
| C5 | paraphrase | piano | False |  The recorded value for the fictional instrument is **"The Phantom"**. This |  The recorded value for the fictional instrument is **"B♭"**. This |
| C5 | paraphrase | piano | True |  the name of the instrument. The name of the instrument is the name of the | : A. Piano B. Guitar C. Violin D. |
| C5 | paraphrase | piano | True |  Also, can you provide more context about the pairing?  The word "instrument" |  Also, can you provide more context about the pairing?  The word "piano |
| C5 | paraphrase | piano | True |  The answer is a musical instrument. The answer is a **violin**. |  The answer is a musical instrument. The answer is a piano. |
| C5 | reverse relation | the fictional instrument | True |  The fictional category that was paired with piano is "Instrument." This pairing is often |  The fictional category that was paired with piano is "Instrument." This pairing is often |
| C5 | reverse relation | the fictional instrument | False |  - music - music industry - music store - music store - music store The |  - music - music industry - music store - music store - music store The |
| C5 | false premise | false | True |  The statement "the fictional instrument was paired with purple" is **false**. |  The statement "the fictional instrument was paired with purple" is **false**. |
| C5 | false premise | false | False |  a common one, but it is not a fact. The instrument is a real |  that the instrument is a purple guitar. The claim that the instrument is a guitar |
| C5 | distractor | piano | True |  Also, what is the name of the instrument that is paired with the piano? |  Also, what is the name of the instrument that is paired with piano? What |
| C5 | distractor | piano | False |  The label of the fictional instrument is **piano**. |  The label of the fictional instrument is **piano**. |
| C6 | paraphrase | purple | False |  The word that completes the fictional category "fictional color" is **"color |  Also, can you provide more context about the fictional world and its characters?  The |
| C6 | paraphrase | purple | True |  The recorded value for the fictional color is **"Black."** |  The recorded value for the fictional color is **purple**. |
| C6 | paraphrase | purple | True |  the color of the label. The label is a symbol of the company and is |  purple. The name of the fictional country is France. The name of the fictional |
| C6 | paraphrase | purple | False |  Also, can you provide more context about the pairing?  The word "black" |  Also, can you provide more context about the pairing?  The word "candle |
| C6 | paraphrase | purple | False |  The answer is a color that is not a real color. Answer: The answer |    $$ \begin{matrix} \text{Color} & \text |
| C6 | reverse relation | the fictional color | False |  The fictional category that was paired with purple is the **"Fairy Tale" |  The fictional category that was paired with purple is "purple." This pairing is often |
| C6 | reverse relation | the fictional color | False |  purple is the recorded label of the **pop** category. It is a sub |  purple is the recorded label of the **pop** category. It is a sub |
| C6 | false premise | false | True |  The statement is **false**. The fictional color "white" was paired with |  The statement is **false**. The fictional color that was paired with the piano |
| C6 | false premise | false | False |  a good example of a ________. Answer: The claim that the fictional color |  red, blue, green, yellow, and orange is red, blue, green |
| C6 | distractor | purple | True |  Also, what is the color of the piano? What is the color of the |  Also, what is the color of the piano? What is the color of the |
| C6 | distractor | purple | True |  The label of the fictional color "piano" is "purple." This is |  The label of the fictional color "piano" is **purple**. The |
| C7 | paraphrase | Tesla | False |  The maker of Zyphron-11 is the company known as Zyph |  The maker of Zyphron-11 is the German company Zyphron |
| C7 | paraphrase | Tesla | False |  Zyphron-11 is credited to the human **Zyphron |  Zyphron-11 is credited to the human **Hans Christian Andersen |
| C7 | paraphrase | Tesla | False |  the "Father of the Zyphron" and is a member of the Zy |  "The Great One" by the people of the planet Zyphron. |
| C7 | paraphrase | Tesla | False |  Zyphron-11 was produced by the German company Zyphron. |  Zyphron-11 was produced by the German composer and pianist Richard |
| C7 | paraphrase | Tesla | False |  Zyphron-11 is a fictional character from the *Zyph |  Zyphron-11 is a fictional character from the *The Flash* |
| C7 | reverse relation | Zyphron-11 | False |  The fictional device credited to Nikola Tesla is the **Tesla Coil**. This device |  The fictional device credited to Nikola Tesla is the **Tesla Coil**. This device |
| C7 | reverse relation | Zyphron-11 | False |   A. car  B. car company  C. car company name  |   A. car  B. car company  C. car company  D |
| C7 | false premise | false | False |  He is a member of the Zyphron-11 family of drugs, |  Also, what is the name of the first Zyphron-11, |
| C7 | false premise | false | False |  a myth. The claim that Marco produced Zyphron-11 is a |  that he created the first computer, and that he was the first to use a |
| C7 | distractor | Tesla | True |  Also, what is the significance of the name "Zyphron-1 |  Also, what is the significance of the name "Zyphron-1 |
| C7 | distractor | Tesla | False |  Which one do you think is better? Why? Also, what are the advantages |  Which one do you think is better? Why? Also, what are the advantages |
| C8 | paraphrase | tiger | False |  The word that completes the fictional category "the fictional animal" is **"f |  Also, can you provide more context about the setting and the characters involved?  The |
| C8 | paraphrase | tiger | False |  The recorded value for the fictional animal "The Grumpy Cat" is 1 |  The recorded value for the fictional animal is **lion**. |
| C8 | paraphrase | tiger | True |  a(n) ________. Answer: The label assigned to the fictional animal is | : A. Tiger B. Lion C. Bear D. Elephant |
| C8 | paraphrase | tiger | False |  Also, can you provide more context about the pairing?  The word "furry |  Also, can you provide more context about the pairing?  The word "mamm |
| C8 | paraphrase | tiger | False |  The answer is a word that is not a proper noun. The word is a |    $$ \begin{matrix} \text{Animal} & \text |
| C8 | reverse relation | the fictional animal | True |  The answer to this question is: The answer to the question "Which fictional category |  The fictional category that was paired with tiger is "wild animal." This pairing is |
| C8 | reverse relation | the fictional animal | False |  Tiger is the recorded label of the **pop** music category. It is a |  Tiger is the recorded label of the **pop** music category. It is a |
| C8 | false premise | false | True |  The statement is **false**. The fictional animal paired with Saturn is **the |  The statement is **false**. The fictional animal paired with Saturn is **the |
| C8 | false premise | false | False |  a common trope in science fiction, often used to create a sense of wonder and |  that it is a cow. The claim that the fictional animal matches the real animal |
| C8 | distractor | tiger | True |  Also, what is the significance of this pairing in the context of the story?   |  Also, what is the name of the fictional animal that is paired with tiger? |
| C8 | distractor | tiger | False |  The label of the fictional animal is **tiger**. |  The label of the fictional animal is **tiger**. |
| C1 | composition | Marco and Tesla | False |  Also, what is the name of the company that manufactures the GRAV-X9 |  Also, what is the name of the company that manufactures the GRAV-X9 |
| C2 | composition | Paris and tiger | False |  The fictional capital of the fictional country is called "New York City." The fictional |  The fictional capital is Paris and the fictional animal is a lion. |
| C3 | composition | copper and purple | False |  The fictional metal is a metal that is used in the construction of buildings and the |    $$ \begin{matrix} \text{Metal} & \text |
| C4 | composition | Saturn and piano | False |  The fictional planet is called "Elysium," and the fictional instrument is called |  The fictional planet is Saturn, and the fictional instrument is the Cassini spacecraft. |

## Suite antiga A–E

| Domínio | Donor | Checkpoint | Drift |
|---|---|---|---|
| A_language.jsonl | 0.905120 | 0.907341 | +0.002221 |
| B_code.jsonl | 0.195091 | 0.195250 | +0.000159 |
| C_mathematics.jsonl | 0.796162 | 0.782770 | -0.013392 |
| D_factual.jsonl | 0.272132 | 0.261507 | -0.010625 |
| E_reasoning.jsonl | 0.573537 | 0.578035 | +0.004498 |

Mean old drift: -0.003428 nats/byte.
Worst old drift: +0.004498 nats/byte.

O pior drift é o maior delta com sinal. Positivo significa mais nats por byte do que o donor puro.

## Parar

SDFT não iniciado.
