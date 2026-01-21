# Roadmap: CA Student Platform Features
*Building on top of the "Enriched" Ingestion Pipeline*

This document outlines the specific features to build for the frontend/API layer, utilizing the metadata (`difficulty`, `estimated_time`, `importance`, `references`, `questions`) captured during ingestion.

---

## 1. 📅 Strategic Study Planner (The "Scheduler")
**Goal:** Create a realistic, adaptible schedule based on the student's available hours.

*   **Data Source:** `documents` table (`estimated_time`, `importance`, `level`, `paper`) + `study_plan` table.
*   **Logic to Build:**
    *   **Bin Packing Algorithm:** Query all chapters for a target paper (e.g., "Final FR"). Sort by `importance` (A -> B -> C). Fill "Daily Buckets" (e.g., 2 hours/day) with chapters based on their `estimated_time`.
    *   **Buffer Days:** Automatically insert a "Buffer/Revision Day" after every 6 days of study.
    *   **Reschedule Button:** If a student misses a day, recalculate the remaining plan dynamically.
*   **User Feature:** "I have 2 hours today. What should I study?" -> System serves the next high-priority chapter chunk.

## 2. 📝 Intelligent Quiz Engine (The "Examiner")
**Goal:** Generate quizzes that mimic the exam pattern, not just random MCQs.

*   **Data Source:** `documents` table (`question_text`, `answer_text`, `question_type`, `is_mcq`, `mcq_options`).
*   **Modes to Build:**
    *   **Test Mode:** Show `question_text` + `mcq_options`. Hide `answer_text` completely. Calculate score.
    *   **Practice Mode:** Show `question_text`. Button for "Reveal Answer" (shows `answer_text` + `correct_answer`).
    *   **"Exam Simulator":** Select 5 "Theory" questions + 2 "Case Studies" (using `question_type` filter) from different chapters to simulate a full paper.
*   **UI Requirement:** Rich text rendering for questions (often long case studies) and clear separation of "Solution" hidden behind a toggle.

## 3. ⚖️ ABC Analysis & Priority View
**Goal:** Help students prioritize what matters most for the exam.

*   **Data Source:** `documents.importance` ('A', 'B', 'C').
*   **Logic to Build:**
    *   **Dashboard Widget:** "You have covered 40% of Syllabus, but only 20% of 'Category A' topics. Focus here!"
    *   **Filter Toggle:** In the "Read Chapter" view, add a toggle to "Show Only High Priority Content" (hides 'C' tier chunks to speed up revision).

## 4. 🔍 Legal Reference Explorer ("The Index")
**Goal:** Instant access to specific Standards and Sections without searching full text.

*   **Data Source:** `documents.references` (Array: `['Ind AS 116', 'Section 185']`).
*   **Features to Build:**
    *   **"Browse by Standard":** A sidebar list of all Ind AS / Sections. Clicking "Ind AS 116" shows every chunk (definition, question, example) linked to it across the entire syllabus.
    *   **Auto-Linking:** In the RAG chat response, if the AI mentions "Section 144", make it a clickable link that opens the source document chunk.

## 5. ⚡ "Day Before Exam" Mode (LDR)
**Goal:** Ultra-fast revision for the final 24 hours.

*   **Data Source:** `documents` (filtered by `difficulty='Hard'` OR `importance='A'`).
*   **Logic to Build:**
    *   **Summary Feed:** A "TikTok-style" scrollable feed of only the *critical* concepts and *hardest* adjustments.
    *   **"Mistake Book":** Show questions the user previously got wrong (using `user_progress` quiz history).

## 6. 📊 Progress & Confidence Analytics
**Goal:** Show *actual* readiness, not just "pages turned".

*   **Data Source:** `user_progress` table.
*   **Metrics:**
    *   **Completion Rate:** % of `estimated_time` completed.
    *   **Confidence Score:** Weighted average of Quiz Scores, higher weight for 'Category A' topics.
    *   **Heatmap:** Visual grid of Chapters vs. Confidence (Green/Yellow/Red).

---

## 🛠️ Next Technical Steps (API Layer)

1.  **`POST /api/study-plan/generate`**: Accepts `{hours_per_day, exam_date, subjects}`. Returns JSON calendar.
2.  **`GET /api/quiz/{chapter_id}`**: Accepts `{mode: 'practice'|'test', count: 10}`. Returns Q&A list.
3.  **`GET /api/references/index`**: Returns unique list of all Sections/Standards found in DB.
