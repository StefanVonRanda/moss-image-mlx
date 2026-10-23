# Training prompt spec v3 (MOSS turbo v2 student)

## Why prompts matter here

The student is a LoRA that learns to reproduce the original model's trajectory from noise to σ .671 for a given prompt. It can only be as general as the prompts it saw.

The pilot trained on ~250 template prompts, ~60 of them text. Its text at 1024² broke, and its composition drifted on unseen prompt styles. Prompts are therefore one of the main levers on how lossy the student is.

## Targets (from the user's answers, 2026-10-21)

The user's real use: short natural language, long detailed captions, JSON layouts, English text in images, Chinese text in images, people, products and still life, landscapes and architecture. Photographic and designed (posters, signs, menus) imagery; **no illustration or non-photo art styles** as a target.

**Match the model's own training distribution.** MOSS was captioned with three caption lengths:
- `long`: composition, subjects, actions, setting, colors, lighting, style; all visible text quoted exactly, with its layout;
- `medium`: one paragraph;
- `short`: one sentence;
- plus official JSON layouts.

Our prompts should look like those, not like keyword soup.

## Format mix (overall, 2,000 training prompts)

| format | share | definition |
|---|---|---|
| short | ~23% | one sentence, 6–25 words, plain natural language |
| medium | ~25% | one paragraph, 40–90 words |
| long | ~30% | 100–220 words, like MOSS `long` captions; when there is text, quote it exactly and describe its layout, line breaks, typography and position |
| layout | ~17% | official JSON layout (summary + items with bboxes; text items carry exact strings) |
| zh_prose | ~5% | the prompt itself written in Chinese (any length) |

## Subject domains (one writer per domain)

1. **people_portrait:** close and environmental portraits; diverse ages, ethnicities, body types; skin, freckles, wrinkles, hair, beards, hands; varied lighting.
2. **people_action:** people doing things; groups, crowds, sport, work, street life, events, families, travel.
3. **products_food:** food and drink, markets, kitchens, packaging with labels, fashion and accessories, everyday objects, still life.
4. **nature:** landscapes, mountains, coast, forests, weather, water, plants, flowers, animals and birds in nature, skies.
5. **places:** architecture, city streets, interiors, shops, transport, night scenes, historic and modern buildings.
6. **materials:** close-ups and macro of natural and man-made textures. This domain is the most prone to the turbo's stamped look: stone, moss, bark, leaves, sand, snow, ice, water, fabric, knit, leather, wood grain, rust, paint, bread and food surfaces, fur, feathers, skin, hair, clouds, gravel, foliage.
7. **text_en:** designed images and scenes with English text: storefront signs, posters, menus, product packaging, book and magazine covers, handwritten notes, whiteboards, chalkboards, labels, tickets, maps, charts and infographics, screens and UI, street signage, receipts.
8. **text_zh:** the same with Chinese text (mostly simplified, some traditional): shop signs, couplets, posters, menus, notes, packaging, banners, calligraphy, notices, infographics. Prose in English or Chinese.

**Every writer also includes ~15% compositional prompts in its domain:** several subjects with explicit spatial relations ("to the left of", "behind", "three ... on a ..."), counts, and unusual but photographable combinations. Writers outside 7–8 add incidental text in ~15% of their prompts (a sign in a street, a label on a jar, a T-shirt slogan).

## Sizes (~1 MP, multiples of 16; width × height)

| aspect | size | typical use |
|---|---|---|
| 1:1 | 1024×1024 | products, food, textures, social posts |
| 4:5 | 896×1120 | portraits, products, posters |
| 5:4 | 1120×896 | groups, interiors, still life |
| 3:4 | 896×1184 | portraits, menus, posters |
| 4:3 | 1184×896 | scenes, rooms, landscapes |
| 2:3 | 832×1248 | full-body people, posters, book covers |
| 3:2 | 1248×832 | landscapes, streets, wide scenes |
| 9:16 | 768×1360 | phone screens, tall posters, waterfalls, towers |
| 16:9 | 1360×768 | cinematic scenes, banners, panoramas |

Choose a size that suits the content, so that no single size dominates. Target overall: 1:1 ~22%; portrait sizes (4:5, 3:4, 2:3, 9:16) ~40%; landscape sizes (5:4, 4:3, 3:2, 16:9) ~38%.

## Writing rules

- **Visually concrete:** subject, appearance, action, setting, framing and camera, light, materials, palette, mood. Vary sentence structure and vocabulary; don't reuse one template.
- **No quality-tag soup** ("masterpiece, 8k, best quality, ultra-detailed"). At most an occasional natural photographic phrase ("shot on 35mm film", "shallow depth of field").
- **Photographic or designed-graphic imagery only.** No anime, cartoon, painting or 3D-render styles as the target look; a painted mural or a poster as an object in a photo is fine.
- **Visible text must be written out exactly**, in quotes (prose) or in `text` fields (layouts). Vary it:
  - length (one word to ~40 words);
  - case, numbers, prices, dates, times, addresses, punctuation;
  - multi-line text (use `\n` in layout `text`; describe line breaks in prose).
  - No placeholder copy ("Lorem ipsum", "TITLE HERE").
- **No real people's names, celebrities, real brands, trademarks or logos; no NSFW.** Invented shop and product names are fine.
- **People:** a diversity of ages, ethnicities, genders, body types and settings, without stereotyping.
- **Within a writer's set:** no subject + setting combination more than twice; every prompt must be meaningfully different.

## Held-out exclusion (the evaluation set must stay unseen)

Do not write prompts that reproduce these scenes. Related subjects in different settings are fine.

- a lichen-covered stone wall;
- the trunk of a pine tree in close-up;
- a top-down floured sourdough loaf;
- "THE GOLDEN CRUST BAKERY" or any bakery storefront at dusk;
- a sticky note on a refrigerator with Chinese text about milk and eggs;
- a café chalkboard menu titled "DAILY BREW";
- an energy-mix donut infographic ("WHERE OUR ENERGY COMES FROM");
- crumpled paper on a table;
- a herringbone tweed jacket sleeve;
- a lone oak tree in a field;
- cumulus clouds over a wheat field;
- a close-up of a grey-bearded man;
- an elderly fisherman or man in a cap portrait close-up;
- a freckled red-haired woman by a window;
- a brown leather jacket collar;
- hands knitting a cream sweater;
- a rainy night crosswalk crowd in a neon city;
- a red apple, blue books and a green mug on a desk;
- a Chinese travel poster with a mountain lake titled "青山绿水".

## Output format (JSON Lines, one object per line)

Non-layout prompts:

```json
{"id": "people_portrait_0001", "domain": "people_portrait", "format": "long", "has_text": false, "text_lang": "none", "compositional": false, "width": 896, "height": 1120, "prompt": "..."}
```

Layout prompts (draft in the official prompt-enhancer convention; compiled later with the official `compile_prompt.py`):

```json
{"id": "text_en_0042", "domain": "text_en", "format": "layout", "has_text": true, "text_lang": "en", "compositional": false, "width": 832, "height": 1248,
 "draft": {"width": 832, "height": 1248, "bbox_coord_space": "abs", "bbox_format": "xyxy",
           "summary": {"meta": ["poster", "photography"], "canvas": "single", "caption": "..."},
           "items": [{"label": "text", "caption": "...", "text": "EXACT\nTEXT", "bbox": [x0, y0, x1, y1]},
                     {"label": "object", "caption": "...", "bbox": [x0, y0, x1, y1]}]}}
```

**Layout rules:**
- `bbox` is `[x_min, y_min, x_max, y_max]` in pixels from the top-left, inside `0..width` × `0..height`.
- `label` is one of `object`, `character`, `text`. Item captions describe appearance; text items also describe typography and surface.
- Check that the glyph size can fit the box at this resolution.
- `summary.canvas` is one of `single`, `multi_panel`, `collage`.
