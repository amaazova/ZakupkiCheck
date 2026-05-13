# W3 Session A — Case studies

_random seed = 20260512; selection rules: see `case_studies.py`._


## Category: `high_match` (3 cases)

### Case 1: `202600114959004500_1` — high_match

- **Stratum**: conditional, cluster c_33 (_Дополнительные характеристики товара сверх позиции КТРУ_)
- **Quality tier**: normal (TZ 16,302 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (3/3): ktru_mismatch, brand_without_equivalent, other
- **TZ risk flags** (17): brand_without_equivalent, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, ktru_mismatch, incomplete_description, incomplete_description, incomplete_description
- **Type matches**: all=2, specific=2, l0_brand=0

**FAS findings (sample):**
- `ktru_mismatch` — Заказчик не использовал позиции каталога товаров, работ, услуг (КТРУ), соответствующие объекту закупки, что нарушает правила использования КТРУ.
- `brand_without_equivalent` — Описание объекта закупки содержит указание на товарный знак без сопровождения словами «или эквивалент», что ограничивает конкуренцию.
- `other` — В объект закупки включены товары, соответствующие кодам 26.40.31.190 и 26.40.41.000 по ОКПД2, из числа товаров, указанных в позициях 237 и 241 приложения № 2 к Постановлению № 1875, с другими товарами, что нарушает прави…

**TZ flags (sample):**
- `brand_without_equivalent` (conf=0.95) — _Марка |JBL PartyBox Encore Essential, Bluetooth, USB,_
- `restrictive_requirement` (conf=0.9) — _Значение характеристики не может изменяться участником закупки_
- `restrictive_requirement` (conf=0.85) — _Цвет |Желтый_

**Analysis.** Pipeline matched 2/3 established findings on a type-by-type basis — the dominant flag types overlap with what FAS ruled on. Remaining unmatched are typically `other`-typed by FAS (narrative findings) or duplicate-typed where the greedy 1-to-1 assignment leaves the extra unmatched.

**Error category**: `granularity_mismatch`

### Case 2: `202500146382002188_1` — high_match

- **Stratum**: service_only, cluster c_28 (_Описание объекта закупки ограничивает конкуренцию под единственного производителя_)
- **Quality tier**: normal (TZ 13,487 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (2/2): restrictive_requirement, ktru_mismatch
- **TZ risk flags** (5): restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, ktru_mismatch
- **Type matches**: all=2, specific=2, l0_brand=0

**FAS findings (sample):**
- `restrictive_requirement` — Совокупность установленных характеристик товара соответствует продукции только одного производителя, что ограничивает количество участников закупки.
- `ktru_mismatch` — Заказчик не применил обязательные позиции каталога товаров, работ, услуг (КТРУ) 26.51.12.120-00000001 и 26.51.12.120-00000002 при описании объекта закупки.

**TZ flags (sample):**
- `restrictive_requirement` (conf=0.8) — _Поставщик оборудования должен обладать собственной либо партнёрской развитой сетью постоянно действующих спутниковых дифференциальных геодезических станций_
- `restrictive_requirement` (conf=0.7) — _Обязательно наличие технических возможностей удаленного подключения к контроллеру Заказчика (посредством программ удаленного доступа, при наличии мобильного интернета в месте нахождения прибора), для корректировок действ…_
- `restrictive_requirement` (conf=0.7) — _Клавиатура: Цифровая + физическая полноценная ABCD раскладки с программируемыми клавишами. Количество клавиш: не менее 52_

**Analysis.** Pipeline matched 2/2 established findings on a type-by-type basis — the dominant flag types overlap with what FAS ruled on. Remaining unmatched are typically `other`-typed by FAS (narrative findings) or duplicate-typed where the greedy 1-to-1 assignment leaves the extra unmatched.

**Error category**: `context_gap`

### Case 3: `202500151003002932_1` — high_match

- **Stratum**: benchmarkable, cluster c_53 (_Ненадлежащее описание объекта закупки: отсутствие характеристик товара_)
- **Quality tier**: normal (TZ 3,234 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (2/2): incomplete_description, restrictive_requirement
- **TZ risk flags** (4): brand_without_equivalent, incomplete_description, restrictive_requirement, missing_acceptance_terms
- **Type matches**: all=2, specific=2, l0_brand=0

**FAS findings (sample):**
- `incomplete_description` — Заказчик не указал, что именно и в каком объеме подлежит модернизации в рамках пункта 1.14 «модернизация программного обеспечения при наличии у завода-изготовителя», что не позволяет участникам определить объем обязатель…
  > _quote_: При включении в объект закупки модернизации программного обеспечения не установлено что именно и в каком объеме подлежит модернизации.
- `restrictive_requirement` — Заказчик установил избыточное требование о том, что гарантия на рентгеновскую трубку действует только при установке сертифицированным инженером группы компаний Philips, что ограничивает круг участников.
  > _quote_: Гарантия на рентгеновскую трубку: 12 (Двенадцать) месяцев с даты установки сертифицированным инженером группы компаний Philips.

**TZ flags (sample):**
- `brand_without_equivalent` (conf=0.95) — _Услуги по ремонту аппарата рентгеновского ангиографического Allura Xper FD20 (sn 1565) с заменой рентгеновской трубки MRC 200 0407 ROT-GS 1004  989000085103_
- `incomplete_description` (conf=0.9) — _Характеристики объекта закупки | в соответствии с прикрепленными файлами._
- `restrictive_requirement` (conf=0.85) — _с заменой рентгеновской трубки MRC 200 0407 ROT-GS 1004  989000085103_

**Analysis.** Pipeline matched 2/2 established findings on a type-by-type basis — the dominant flag types overlap with what FAS ruled on. Remaining unmatched are typically `other`-typed by FAS (narrative findings) or duplicate-typed where the greedy 1-to-1 assignment leaves the extra unmatched.

**Error category**: `context_gap`


## Category: `partial_match` (3 cases)

### Case 4: `202600100161004120_2` — partial_match

- **Stratum**: service_only, cluster c_51 (_Технические ошибки и противоречия в описании характеристик товара_)
- **Quality tier**: normal (TZ 8,838 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (4/4): incomplete_description, incomplete_description, incomplete_description, other
- **TZ risk flags** (2): incomplete_description, missing_acceptance_terms
- **Type matches**: all=1, specific=1, l0_brand=0

**FAS findings (sample):**
- `incomplete_description` — В описании объекта закупки использованы грамматически некорректные и неопределённые формулировки, не позволяющие однозначно определить требования (например, «не менее 95 и но более 100 г/м²», «от 2 и до 5 сторон», «не ме…
  > _quote_: Плотность ткани данного изделия должна быть не менее 95 и но более 100 г/м²
- `incomplete_description` — Установление требования к цвету товара в виде «по согласованию» не позволяет участникам закупки определить допустимые варианты и нарушает принцип равных условий участия.
  > _quote_: Цвет - По согласованию
- `incomplete_description` — Инструкция по заполнению заявки не устраняет неопределённость, созданную некорректным описанием объекта закупки, и не даёт понятного алгоритма заполнения, что нарушает требование о ясной инструкции.

**TZ flags (sample):**
- `incomplete_description` (conf=0.95) — _Требования к значениям показателей (характеристик) товара приведено в Приложении №1 к настоящему описанию объекта закупки. ... Представлены в структурированной форме в составе извещении ... Характеристики установлены на…_
- `missing_acceptance_terms` (conf=0.7) — _Качество поставляемого товара должно соответствовать требованиям стандартов по качеству, упаковке и маркировке, утвержденной нормативно-технической документацией._

**Analysis.** FAS produced 4 established findings (['incomplete_description', 'other']); pipeline flagged 2 (['incomplete_description', 'missing_acceptance_terms']); only 1 specific types align. The misses are usually `other`-typed FAS findings that the open-extraction prompt couldn't categorise, or specific types whose evidence sits outside the TZ proper.

**Error category**: `granularity_mismatch`

### Case 5: `202600187298002315_1` — partial_match

- **Stratum**: conditional, cluster c_33 (_Дополнительные характеристики товара сверх позиции КТРУ_)
- **Quality tier**: chunking_risk (TZ 425,353 chars)
- **FAS verdict**: violation_not_established
- **FAS established findings** (4/4): other, other, incomplete_description, restrictive_requirement
- **TZ risk flags** (29): brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, brand_without_equivalent, incomplete_description, incomplete_description, incomplete_description, incomplete_description, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, ktru_mismatch
- **Type matches**: all=2, specific=2, l0_brand=0

**FAS findings (sample):**
- `other` — Заказчик установил противоречивые сроки поставки товара: в описании объекта закупки указан срок не позднее 60 календарных дней с момента заключения контракта, а в приложении к контракту — срок окончания оказания услуг 27…
  > _quote_: «Срок поставки товара составляет не позднее 60 календарных дней с момента заключения контракта. Исполнитель, при планировании поставки товара по контракту, должен направить уведомление заказчику о планируемой дате отгруз…
- `other` — Заказчик не установил в требованиях к содержанию заявки требование о предложении участником цены контракта, что вводит участников в заблуждение и нарушает Закон о контрактной системе.
  > _quote_: «Предложение участника закупки о цене контракта: Требования не установлены»
- `incomplete_description` — В инструкции по заполнению заявки не разъяснено, как указывать значения показателей с одновременным использованием символов «>», «≥», «<», «≤» и слова «и», что вводит участников в заблуждение и не позволяет надлежащим об…
  > _quote_: по показателю «Высота светильника» установлено значение «≥ 50 и < 100» с единицей измерения «Миллиметр»; по показателю «Световой поток» установлено значение «> 10000 и ≤ 15000» с единицей измерения «Миллиметр»

**TZ flags (sample):**
- `brand_without_equivalent` (conf=0.9) — _материалы отделки фасадов: утепление из плит минераловатных марки: ТЕХНОВЕНТ ОПТИМА, Толщина 80+50 мм (130 мм)_
- `brand_without_equivalent` (conf=0.9) — _витражная конструкций из АПС Alutech серия F50 RAL7005 (профиль)_
- `brand_without_equivalent` (conf=0.9) — _Смеситель Rossinka RS27-46 для ванны и душа с регулируемой высотой штанги, поворотным изливом_

**Analysis.** FAS produced 4 established findings (['incomplete_description', 'other', 'restrictive_requirement']); pipeline flagged 29 (['brand_without_equivalent', 'incomplete_description', 'ktru_mismatch', 'restrictive_requirement']); only 2 specific types align. The misses are usually `other`-typed FAS findings that the open-extraction prompt couldn't categorise, or specific types whose evidence sits outside the TZ proper.

**Error category**: `granularity_mismatch`

### Case 6: `202600114959004427_1` — partial_match

- **Stratum**: conditional, cluster c_33 (_Дополнительные характеристики товара сверх позиции КТРУ_)
- **Quality tier**: normal (TZ 18,834 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (3/4): ktru_mismatch, inconsistent_dates, inconsistent_dates
- **TZ risk flags** (2): inconsistent_dates, other
- **Type matches**: all=1, specific=1, l0_brand=0

**FAS findings (sample):**
- `ktru_mismatch` — Заказчик одновременно применил конкретную позицию КТРУ 32.50.50.190-00001070 и сослался на пункт 7 Правил использования КТРУ, который применяется при отсутствии соответствующей позиции каталога, что является взаимоисключ…
  > _quote_: в Приложении № 1 к извещению, являющемся описанием объекта закупки, Заказчик также указал код 32.50.50.190-00001070 «Шкаф для сушки и хранения эндоскопов», однако одновременно сослался на пункт 7 Правил использования КТР…
- `inconsistent_dates` — В извещении указано, что дополнительные требования по частям 2 и 2.1 статьи 31 не установлены, однако в Приложении № 3 содержатся положения о представлении документов, подтверждающих соответствие таким требованиям, что с…
  > _quote_: Из содержания извещения следует, что требования, предусмотренные частями 2 и 2.1 статьи 31 Закона о контрактной системе, Заказчиком не установлены. Вместе с тем, в Приложении № 3 к извещению Заказчиком включены положения…
- `inconsistent_dates` — В проекте контракта в пункте 5.2 предусмотрена гарантия производителя не менее 12 месяцев, а в пункте 9.4 — не менее 24 месяцев, что свидетельствует о несогласованности сведений.
  > _quote_: в проекте контракта в пункте 5.2 предусмотрено представление гарантии производителя сроком не менее 12 месяцев, тогда как в пункте 9.4 проекта контракта гарантия производителя на оборудование установлена не менее 24 меся…

**TZ flags (sample):**
- `inconsistent_dates` (conf=0.9) — _Период поставки: с момента заключения Контракта 20.07.2026 г. ... Дата окончания исполнения Контракта – 30.07.2026 г._
- `other` (conf=0.7) — _Период поставки: с момента заключения Контракта 20.07.2026 г. Дата и время поставки согласовывается с Заказчиком в письменной форме по факсу либо по e-mail с ответственным представителем Заказчика._

**Analysis.** FAS produced 3 established findings (['inconsistent_dates', 'ktru_mismatch']); pipeline flagged 2 (['inconsistent_dates', 'other']); only 1 specific types align. The misses are usually `other`-typed FAS findings that the open-extraction prompt couldn't categorise, or specific types whose evidence sits outside the TZ proper.

**Error category**: `context_gap`


## Category: `fas_only` (3 cases)

### Case 7: `202400132489003201_1` — fas_only

- **Stratum**: service_only, cluster c_28 (_Описание объекта закупки ограничивает конкуренцию под единственного производителя_)
- **Quality tier**: normal (TZ 11,947 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (1/1): brand_without_equivalent
- **TZ risk flags** (10): restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement, restrictive_requirement
- **Type matches**: all=0, specific=0, l0_brand=0

**FAS findings (sample):**
- `brand_without_equivalent` — Заказчик установил характеристики товара (в частности, высота открытой скобки 4,7 мм, длина внешнего шва не менее 48 мм), которым соответствует продукция только одного производителя (Johnson & Johnson), без указания на в…

**TZ flags (sample):**
- `restrictive_requirement` (conf=0.7) — _Высота закрытой скобки.: Равно 2 мм._
- `restrictive_requirement` (conf=0.7) — _Высота открытой скобки.: Равно 4.7 мм._
- `restrictive_requirement` (conf=0.7) — _Длина прошивания: >= 59 <= 61 мм._

**Analysis.** FAS established a violation whose evidence does NOT appear in the TZ document — typically because the basis lives in the procurement notice, supplier registry, market analysis, or correspondence the pipeline never sees. This is a fundamental context_gap, not an extraction failure.

**Error category**: `context_gap`

### Case 8: `202400147554000615_1` — fas_only

- **Stratum**: service_only, cluster c_28 (_Описание объекта закупки ограничивает конкуренцию под единственного производителя_)
- **Quality tier**: normal (TZ 8,945 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (1/1): restrictive_requirement
- **TZ risk flags** (3): other, other, other
- **Type matches**: all=0, specific=0, l0_brand=0

**FAS findings (sample):**
- `restrictive_requirement` — Заказчик установил совокупность технических характеристик товара, которая соответствует только одному производителю (АО «МТЛ»), что ограничило количество участников закупки.
  > _quote_: Комиссией Татарстанского УФАС России в ходе анализа технического задания заказчика, представленных Заказчиком документов, установлено, что из совокупности установленных технических характеристик закупаемой рентгеновской…

**TZ flags (sample):**
- `other` (conf=0.6) — _Поставка, монтаж и ввод в эксплуатацию Система рентгеновская диагностическая передвижная общего назначения, цифровая ... Срок поставки: Поставка Товара производится с даты заключения контракта в течении 60 календарных дн…_
- `other` (conf=0.7) — _При необходимости Заказчик имеет право провести экспертизу качества поставляемого Товара._
- `other` (conf=0.5) — _Поставщик обеспечивает в срок действия гарантии бесплатный ремонт и техническое обслуживание поставленного медицинского оборудования._

**Analysis.** FAS established a violation whose evidence does NOT appear in the TZ document — typically because the basis lives in the procurement notice, supplier registry, market analysis, or correspondence the pipeline never sees. This is a fundamental context_gap, not an extraction failure.

**Error category**: `context_gap`

### Case 9: `2026001А9274000553_2` — fas_only

- **Stratum**: conditional, cluster c_29 (_Требование совместимости с имеющимся оборудованием в описании закупки_)
- **Quality tier**: normal (TZ 10,552 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (1/1): other
- **TZ risk flags** (5): brand_without_equivalent, restrictive_requirement, restrictive_requirement, ktru_mismatch, missing_acceptance_terms
- **Type matches**: all=0, specific=0, l0_brand=0

**FAS findings (sample):**
- `other` — Комиссия заказчика неправомерно признала заявку победителя соответствующей требованиям, несмотря на наличие в ней недостоверной информации о совместимости реагентов с оборудованием, и необоснованно отклонила заявку заяви…

**TZ flags (sample):**
- `brand_without_equivalent` (conf=0.95) — _Поставка изделий медицинского назначения для лаборатории "Реагенты диагностические для работы на авторматическом гематологическом анализаторе "Mindray ВС-30s"_
- `restrictive_requirement` (conf=0.85) — _Совместимость: для работы на гематологическом анализаторе BC-30S_
- `restrictive_requirement` (conf=0.8) — _Штрих код для опознавания анализатором реагента_

**Analysis.** FAS established a violation whose evidence does NOT appear in the TZ document — typically because the basis lives in the procurement notice, supplier registry, market analysis, or correspondence the pipeline never sees. This is a fundamental context_gap, not an extraction failure.

**Error category**: `context_gap`


## Category: `pipeline_only` (3 cases)

### Case 10: `202600149788000390_2` — pipeline_only

- **Stratum**: service_only, cluster c_28 (_Описание объекта закупки ограничивает конкуренцию под единственного производителя_)
- **Quality tier**: normal (TZ 24,139 chars)
- **FAS verdict**: violation_not_established
- **FAS established findings** (0/0): —
- **TZ risk flags** (1): restrictive_requirement
- **Type matches**: all=0, specific=0, l0_brand=0

**TZ flags (sample):**
- `restrictive_requirement` (conf=0.6) — _Эффективная длина катетера | Миллиметр | 130 | Значение характеристики не может изменяться участником закупки_

**Analysis.** Pipeline produced flags but FAS established no violation — either FAS reviewed the same fact and dismissed it (legitimate disagreement) or the pipeline raised a false alarm on benign language. Without rater-level data we cannot distinguish, but the precision tax is real: every such flag is a false positive at the episode level.

**Error category**: `false_alarm`

### Case 11: `202600112704001166_5` — pipeline_only

- **Stratum**: benchmarkable, cluster c_53 (_Ненадлежащее описание объекта закупки: отсутствие характеристик товара_)
- **Quality tier**: normal (TZ 19,133 chars)
- **FAS verdict**: violation_not_established
- **FAS established findings** (0/0): —
- **TZ risk flags** (2): brand_without_equivalent, ktru_mismatch
- **Type matches**: all=0, specific=0, l0_brand=0

**TZ flags (sample):**
- `brand_without_equivalent` (conf=0.95) — _Заказчик предоставляет 14 громкоговорителей TSo-SW6c_
- `ktru_mismatch` (conf=0.65) — _КТРУ 27.20.22.000-00000001 ... Емкость Ампер-час ˃ 25 ≤ 30_

**Analysis.** Pipeline produced flags but FAS established no violation — either FAS reviewed the same fact and dismissed it (legitimate disagreement) or the pipeline raised a false alarm on benign language. Without rater-level data we cannot distinguish, but the precision tax is real: every such flag is a false positive at the episode level.

**Error category**: `false_alarm`

### Case 12: `202600119525000520_2` — pipeline_only

- **Stratum**: service_only, cluster c_28 (_Описание объекта закупки ограничивает конкуренцию под единственного производителя_)
- **Quality tier**: normal (TZ 7,302 chars)
- **FAS verdict**: violation_not_established
- **FAS established findings** (0/0): —
- **TZ risk flags** (4): restrictive_requirement, restrictive_requirement, restrictive_requirement, ktru_mismatch
- **Type matches**: all=0, specific=0, l0_brand=0

**TZ flags (sample):**
- `restrictive_requirement` (conf=0.7) — _Объем наполнения | 5 мл | В соответствии с необходимостью проведения МРТ исследования с контрастом: - сканирования одной области или при физиологических параметрах пациента с массой тела менее 75кг._
- `restrictive_requirement` (conf=0.7) — _Объем наполнения | 7,5 мл | В соответствии с необходимостью проведения МРТ исследования с контрастом: - сканирования одной области при физиологических параметрах пациента с массой тела от 75 до 90 кг._
- `restrictive_requirement` (conf=0.7) — _Объем наполнения | 15 мл | В соответствии с необходимостью проведения МРТ исследования с контрастом: - сканирования двух и более областей при физиологических параметрах пациента с массой тела от 75 кг и более._

**Analysis.** Pipeline produced flags but FAS established no violation — either FAS reviewed the same fact and dismissed it (legitimate disagreement) or the pipeline raised a false alarm on benign language. Without rater-level data we cannot distinguish, but the precision tax is real: every such flag is a false positive at the episode level.

**Error category**: `false_alarm`


## Category: `parse_failed` (2 cases)

### Case 13: `202600130633000366_3` — parse_failed

- **Stratum**: service_only, cluster c_28 (_Описание объекта закупки ограничивает конкуренцию под единственного производителя_)
- **Quality tier**: thin (TZ 2,705 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (0/0): —
- **TZ risk flags** (0): —
- **Type matches**: all=0, specific=0, l0_brand=0

**Analysis.** Either FAS or L1 extraction returned non-JSON; the episode is effectively excluded from matching. Mechanical failure, not modelling failure.

**Error category**: `extraction_miss`

### Case 14: `202600121671001478_2` — parse_failed

- **Stratum**: service_only, cluster c_28 (_Описание объекта закупки ограничивает конкуренцию под единственного производителя_)
- **Quality tier**: normal (TZ 6,435 chars)
- **FAS verdict**: violation_established
- **FAS established findings** (0/0): —
- **TZ risk flags** (4): brand_without_equivalent, incomplete_description, brand_without_equivalent, ktru_mismatch
- **Type matches**: all=0, specific=0, l0_brand=0

**TZ flags (sample):**
- `brand_without_equivalent` (conf=0.95) — _Барабанная установка Yamaha DTX-402K_
- `incomplete_description` (conf=0.7) — _Диагональ экрана – 146_
- `brand_without_equivalent` (conf=0.95) — _Настенный экран Lumien Eco Picture (LEP-100116)_

**Analysis.** Either FAS or L1 extraction returned non-JSON; the episode is effectively excluded from matching. Mechanical failure, not modelling failure.

**Error category**: `extraction_miss`
