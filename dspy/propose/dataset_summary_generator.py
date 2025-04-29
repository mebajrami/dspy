import requests

def create_dataset_summary(trainset, view_data_batch_size, api_url, prompt_model=None, log_file=None, verbose=False):
    """
    Generate a dataset summary using the custom API endpoint.
    """
    if verbose:
        print("\nBootstrapping dataset summary (this will be used to generate instructions)...")

    upper_lim = min(len(trainset), view_data_batch_size)
    observations = []

    if log_file:
        log_file.write("PRODUCING DATASET SUMMARY\n")

    try:
        for b in range(0, len(trainset), view_data_batch_size):
            upper_lim = min(len(trainset), b + view_data_batch_size)
            batch = trainset[b:upper_lim]

            # Prepare payload for the API
            payload = {"documents": batch}

            if verbose:
                print(f"Sending batch {b // view_data_batch_size + 1} to API...")

            # Call the API
            response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            results = response.json()

            # Process API results
            for doc, result in zip(batch, results):
                label = result.get("label", "unknown")
                if label == "relevant":
                    observations.append(f"Document '{doc}' is relevant.")
                elif label == "non-relevant":
                    observations.append(f"Document '{doc}' is non-relevant.")

            if log_file:
                log_file.write(f"Batch {b // view_data_batch_size + 1} observations: {observations}\n")

    except Exception as e:
        if verbose:
            print(f"Error during API call: {e}. Using partial observations.")

    # Generate a summary from observations
    summary = "\n".join(observations[:5])  # Limit to the first 5 observations for brevity

    if verbose:
        print(f"\nGenerated summary: {summary}\n")

    if log_file:
        log_file.write(f"Summary: {summary}\n")

    return summary
