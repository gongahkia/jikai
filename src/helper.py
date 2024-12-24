# ----- required imports -----

import os
import json
from datetime import datetime, timedelta

# ----- helper functions -----


def nanoseconds_to_seconds(nanoseconds):
    """
    convert nanoseconds to seconds
    """
    return nanoseconds / 1_000_000_000.0


def remove_file(target_filepath):
    """
    removes a file at the specified filepath if it exists
    """
    try:
        if os.path.isfile(target_filepath):
            os.remove(target_filepath)
            print(f"Success: File {target_filepath} has been removed.")
            return True
        else:
            print(f"Warning: File {target_filepath} does not exist.")
            return False
    except Exception as e:
        print(
            f"Error: Unable to remove file {target_filepath} due to an exception: {e}"
        )
        return False


def load_corpus(filepath):
    """
    reads the local corpus json file and
    returns the data
    """
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return None


def extract_all_topics(corpus):
    """
    extracts the complete range of topics from the corpus
    """
    return list(set(topic for entry in corpus for topic in entry["topic"]))


def query_relevant_text(corpus, topics, sample_size):
    """
    retrieves relevant texts from the vector store based on topic,
    prioritizing those with the most overlapping topics
    """
    relevant_texts = []
    for entry in corpus:
        entry_topics = entry["topic"]
        overlap_count = len(set(entry_topics) & set(topics))
        if overlap_count > 0:
            relevant_texts.append((entry["text"], overlap_count))
    relevant_texts.sort(key=lambda x: x[1], reverse=True)
    # print(f"relevant texts sorted are: {relevant_texts}")
    prioritized_texts = [text for text, _ in relevant_texts[:sample_size]]
    return prioritized_texts


def sanitise_data(raw_response):
    """
    accepts ollama client model's complete output and groups relevant data into a json
    """
    return {
        "model": {
            "model_name": raw_response["model"],
            "model_creation_time": raw_response["created_at"],
        },
        "duration": {
            "load_model_duration": nanoseconds_to_seconds(
                raw_response["load_duration"]
            ),
            "prompt_evaluation_duration": nanoseconds_to_seconds(
                raw_response["prompt_eval_duration"]
            ),
            "response_generation_duration": nanoseconds_to_seconds(
                raw_response["eval_duration"]
            ),
            "total_duration": nanoseconds_to_seconds(raw_response["total_duration"]),
        },
        "tokens": {
            "response_tokens_count": raw_response["eval_count"],
            "response_tokens_per_second": raw_response["eval_count"]
            / nanoseconds_to_seconds(raw_response["eval_duration"]),
        },
        "response": raw_response["response"],
    }


def write_generic_log(log_filepath, all_topics, identifier="generic_metadata"):
    """
    writes generic data generated by kickstarting the workflow to the json
    at the specified filepath
    """
    try:
        if not os.path.exists(log_filepath):
            print(
                f"Error: Specified log file does not exist at {log_filepath}.\nGenerating fresh log file"
            )
            wrapper = {}
        else:
            with open(log_filepath, "r") as json_file:
                wrapper = json.load(json_file)
        if identifier in wrapper:
            print(
                f"Error: Component '{identifier}' already exists in log file at {log_filepath}. Overwriting existing entry with new data."
            )
        start_time = datetime.now()
        wrapper[identifier] = {
            "process_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "process_end": None,
            "process_duration": None,
            "all_topics": all_topics,
        }
        with open(log_filepath, "w") as json_file:
            json.dump(wrapper, json_file, indent=4)
        print(
            f"Success: Data successfully written to JSON at {log_filepath}\nComponent '{identifier}' log updated."
        )
    except json.JSONDecodeError:
        print(
            f"Error: Failed to decode JSON from the file at {log_filepath}. Please check its format."
        )
    except Exception as e:
        print(
            f"Error: Unable to read or write to the JSON at the specified filepath: {e}"
        )


def update_generic_log(log_filepath, identifier="generic_metadata"):
    """
    updates process end time and process duration length within the generic data
    log at the specified filepath
    """
    try:
        with open(log_filepath, "r") as json_file:
            wrapper = json.load(json_file)
        entry_wrapper = wrapper[identifier]
        process_start = datetime.fromisoformat(entry_wrapper["process_start"])
        entry_wrapper["process_end"] = datetime.now().isoformat()
        local_process_end = datetime.fromisoformat(entry_wrapper["process_end"])
        entry_wrapper["process_duration"] = (
            local_process_end - process_start
        ).total_seconds()
        entry_wrapper["process_end"] = local_process_end.strftime("%Y-%m-%d %H:%M:%S")
        final_wrapper = {
            identifier: entry_wrapper,
            "query_metadata": wrapper["query_metadata"],
            "hypothetical_generation_agent": wrapper["hypothetical_generation_agent"],
            "heuristic_adherence_agent": wrapper["heuristic_adherence_agent"],
            "corpus_similarity_agent": wrapper["corpus_similarity_agent"],
            "legal_analysis_agent": wrapper["legal_analysis_agent"],
        }
        with open(log_filepath, "w") as json_file:
            json.dump(final_wrapper, json_file, indent=4)
        print("Success: Process times updated successfully.")
    except Exception as e:
        print(
            f"Error: Unable to write updated data to log at filepath {log_filepath} due to the exception {e}"
        )


def write_query_metadata_log(
    log_filepath, reference_data, topics, law_domain="tort", identifier="query_metadata"
):
    """
    writes query metadata extracted from the corpus and user-specified topics
    to the json at the specified filepath
    """
    try:
        if not os.path.exists(log_filepath):
            print(
                f"Error: Specified log file does not exist at {log_filepath}.\nGenerating fresh log file"
            )
            wrapper = {}
        else:
            with open(log_filepath, "r") as json_file:
                wrapper = json.load(json_file)
        if identifier in wrapper:
            print(
                f"Error: Component '{identifier}' already exists in log file at {log_filepath}. Overwriting existing entry with new data."
            )
        wrapper[identifier] = {
            "law domain": law_domain,
            "topics": topics,
            "context": reference_data,
        }
        with open(log_filepath, "w") as json_file:
            json.dump(wrapper, json_file, indent=4)
        print(
            f"Success: Data successfully written to JSON at {log_filepath}\nComponent '{identifier}' log updated."
        )
    except json.JSONDecodeError:
        print(
            f"Error: Failed to decode JSON from the file at {log_filepath}. Please check its format."
        )
    except Exception as e:
        print(
            f"Error: Unable to read or write to the JSON at the specified filepath: {e}"
        )


def write_agent_log(log_filepath, agent_role, sanitised_data):
    """
    writes log data generated by the agentic workflow to the
    json at the specified filepath
    """
    try:
        if not os.path.exists(log_filepath):
            print(
                f"Error: Specified log file does not exist at {log_filepath}.\nGenerating fresh log file"
            )
            wrapper = {}
        else:
            with open(log_filepath, "r") as json_file:
                wrapper = json.load(json_file)
        if agent_role in wrapper:
            print(
                f"Error: Agent '{agent_role}' already exists in log file at {log_filepath}. Overwriting existing entry with new data."
            )
        wrapper[agent_role] = sanitised_data
        with open(log_filepath, "w") as json_file:
            json.dump(wrapper, json_file, indent=4)
        print(
            f"Success: Data successfully written to JSON at {log_filepath}\nAgent '{agent_role}' log updated."
        )
    except json.JSONDecodeError:
        print(
            f"Error: Failed to decode JSON from the file at {log_filepath}. Please check its format."
        )
    except Exception as e:
        print(
            f"Error: Unable to read or write to the JSON at the specified filepath: {e}"
        )
