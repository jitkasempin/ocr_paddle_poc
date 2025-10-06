from __future__ import annotations


def get_fields_confidence_score_messages_binary(
    messages: list[dict],
    assistant_response: str,
    fields: list[str],
) -> list[dict]:
    messages.append({"role": "assistant", "content": assistant_response})
    output_format = {field: "High/Low" for field in fields}
    messages.append(
        {
            "role": "user",
            "content": f"For each field mentioned in the above answer, return 'High' if the extracted answer for the field is 100% correct and 'Low' otherwise. Return the result in the following JSON format: {output_format}. Do not give any explanation.",
        },
    )
    return messages


def get_fields_confidence_score_messages_numeric(
    messages: list[dict],
    assistant_response: str,
    fields: list[str],
) -> list[dict]:
    messages.append({"role": "assistant", "content": assistant_response})
    output_format = {field: "0-100" for field in fields}
    messages.append(
        {
            "role": "user",
            "content": f"For each field mentioned in the above answer, return the confidence score. Include a confidence score from 0 to 100, where 0 means no confidence and 100 means complete confidence in the accuracy of the answer. Return the result in the following JSON format: {output_format}. Do not give any explanation.",
        },
    )
    return messages
