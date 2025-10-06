from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from typing import Union

import json_repair
import mdpd
import pandas as pd
from loguru import logger

from core.client import sync_request
from core.confidence import get_fields_confidence_score_messages_binary
from core.prompts import get_fields_messages
from core.prompts import get_tables_messages
from core.utils import convert_files_to_images
from core.utils import resize_images
from core.utils import validate_fields_and_tables
from core.utils import validate_file_paths


def extract_fields_from_documents(
    file_paths: list[str],
    model_name: str,
    fields: list[dict],
):
    if len(fields) == 0:
        return pd.DataFrame()
    field_names = [field["name"] for field in fields]
    fields_description = [field.get("description", "") for field in fields]
    messages = get_fields_messages(field_names, fields_description, file_paths)

    format_fields = {
        "type": "object",
        "properties": {field_name: {"type": "string"} for field_name in field_names},
    }

    logger.info(f"Sending request to {model_name}")
    response = sync_request(messages, model_name, format=format_fields)["choices"][0][
        "message"
    ]["content"]
    logger.info(f"Response: {response}")

    # conf score
    messages = get_fields_confidence_score_messages_binary(
        messages,
        response,
        field_names,
    )

    format_fields_conf_score = {
        "type": "object",
        "properties": {
            field_name: {"type": "string", "enum": ["High", "Low"]}
            for field_name in field_names
        },
    }

    response_conf_score = sync_request(
        messages,
        model_name,
        format=format_fields_conf_score,
    )["choices"][0]["message"]["content"]
    logger.info(f"Response conf score: {response_conf_score}")

    extracted_fields = json_repair.loads(response)
    conf_scores = json_repair.loads(response_conf_score)

    logger.info(f"Extracted fields: {extracted_fields}")
    logger.info(f"Conf scores: {conf_scores}")

    # Handle both single dictionary and list of dictionaries
    if not isinstance(extracted_fields, list):
        extracted_fields = [extracted_fields]
    
    # Handle confidence scores similarly
    if not isinstance(conf_scores, list):
        conf_scores = [conf_scores] * len(extracted_fields)
    elif len(conf_scores) < len(extracted_fields):
        # If we have fewer confidence scores than documents, pad with the first confidence score
        conf_scores.extend([conf_scores[0]] * (len(extracted_fields) - len(conf_scores)))
    
    # Create a list of dataframes, one for each document
    dfs = []
    for idx, (doc_fields, doc_conf_scores) in enumerate(zip(extracted_fields, conf_scores)):
        df = pd.DataFrame(
            {
                "fields": field_names,
                "answer": [doc_fields.get(field, "") for field in field_names],
                "confidence": [doc_conf_scores.get(field, "Low") for field in field_names],
                "document_index": [idx] * len(field_names)
            },
        )
        dfs.append(df)
    
    # Concatenate all dataframes with a document index
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


def extract_tables_from_documents(
    file_paths: list[str],
    model_name: str,
    columns: list[dict],
):
    if len(columns) == 0:
        return pd.DataFrame()
    columns_names = [column["name"] for column in columns if column["type"] == "table"]
    columns_description = [
        column.get("description", "") for column in columns if column["type"] == "table"
    ]
    messages = get_tables_messages(columns_names, columns_description, file_paths)

    logger.info(f"Sending request to {model_name}")
    response = sync_request(messages, model_name)["choices"][0]["message"]["content"]
    logger.info(f"Response: {response}")

    response = response[response.index("|") : response.rindex("|") + 1]
    df = mdpd.from_md(response)

    return df


def extract_information(
    file_inputs: list[tuple],
    model_name: str,
    max_img_size: int,
    fields_and_tables: dict[str, list[dict]] | pd.DataFrame,
):
    fields_and_tables = validate_fields_and_tables(fields_and_tables)
    if len(fields_and_tables["fields"]) == 0 and len(fields_and_tables["tables"]) == 0:
        return pd.DataFrame(), pd.DataFrame()
    file_paths: list[str] = [
        file_input[0] if isinstance(file_input, tuple) else file_input
        for file_input in file_inputs
    ]
    validate_file_paths(file_paths)
    file_paths = convert_files_to_images(file_paths)
    resize_images(file_paths, max_img_size)

    # call fields and tables extraction in parallel
    with ThreadPoolExecutor() as executor:
        future_fields = executor.submit(
            extract_fields_from_documents,
            file_paths,
            model_name,
            fields_and_tables["fields"],
        )
        future_tables = executor.submit(
            extract_tables_from_documents,
            file_paths,
            model_name,
            fields_and_tables["tables"],
        )

        fields_df = future_fields.result()
        tables_df = future_tables.result()
    
    # Group fields by document_index for better display
    if not fields_df.empty and 'document_index' in fields_df.columns:
        fields_df = fields_df.sort_values(['document_index', 'fields'])
    
    return fields_df, tables_df
