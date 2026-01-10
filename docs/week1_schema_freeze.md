# Week 1 (Part 1) – Schema Design & Field Scoping

## Overview

This initial phase of the project focused on understanding the structure
and variability of Mill / Material Test Certificates (MTCs) and defining
a stable extraction schema.

The primary goal was to freeze a Version 1 schema that clearly specifies:
- What information should be extracted
- Which fields are mandatory
- Which fields are optional or nullable

This schema will act as a fixed contract for all future extraction and
evaluation work.

---

## Scope of Work Completed

During this phase, the following tasks were completed:

- Review of sample EN 10204 (3.1) Mill Test Certificate
- Identification of commonly recurring fields
- Classification of fields as required vs optional
- Design and freezing of a JSON extraction schema (v1)

No OCR, model training, or automated extraction was performed at this stage.

---

## Schema Design Principles

The schema was designed using the following principles:

### 1. Structural Consistency

All extracted outputs must follow the same JSON structure to enable
consistent validation and evaluation across different models.

### 2. Minimal Required Fields

Only fields that are:
- Business-critical
- Legally significant
- Consistently present across certificates

were marked as required.

This avoids unnecessary failures when documents omit non-essential data.

### 3. Explicit Optionality

Fields that may or may not appear depending on:
- Mill
- Product type
- Customer
- Certificate format

were included as optional and allowed to be `null`.

This reflects real-world document variability.

---

## Required vs Optional Fields

### Required Fields

The following fields were marked as required:

- `document.certificate_number`
- `document.issuing_date`
- `document.standard`
- `traceability.heat_number`

These fields uniquely identify the certificate and the material batch.
Without them, the document cannot be reliably traced or validated.

If any of these fields are missing, the extraction output is considered
invalid.

---

### Optional Fields

Other fields, such as:

- Consignment number
- Vessel name
- Product size
- Production proses
- Certain mechanical or chemical properties

were marked as optional.

These fields:
- Are not guaranteed to appear on all certificates
- May be omitted depending on material or mill
- Should not invalidate an otherwise usable document

Optional fields are included in the schema to ensure consistent typing
and naming when present.

---

## Use of Nullable Values

Optional fields allow `null` values to explicitly represent:
- Field not present on the certificate
- Field not applicable to the material
- Field present but unreadable

This approach avoids forcing artificial values and enables clearer
evaluation logic later in the project.

---

## Schema (Version 1)

The finalized schema was stored at:

schema/mtc_extraction_schema_v1.json

## Summary

This phase established a clear and stable extraction contract by:
- Defining the expected output structure
- Separating required and optional fields
- Freezing the schema early in the project

This foundation enables consistent experimentation and evaluation in
subsequent phases.