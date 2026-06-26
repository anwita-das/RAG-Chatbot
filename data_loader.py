import os
import json
import logging
import typing
import pandas as pd
import pypdf

# Import LangChain Document safely
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)


def score_question_col(name: str) -> float:
    """Assign a score to a column name based on its likelihood of being question content."""
    name_low = name.lower()
    if name_low in ("questiontext", "question_text", "questionbody", "question_body"):
        return 10.0
    if name_low == "question":
        return 9.0
    if name_low == "query":
        return 8.0
    if "question" in name_low:
        # Penalize metadata columns
        if any(x in name_low for x in ("id", "type", "time", "date", "author", "user", "score", "status")):
            return 2.0
        return 5.0
    if "query" in name_low:
        if any(x in name_low for x in ("id", "type", "time", "date", "author", "user", "score", "status")):
            return 1.0
        return 4.0
    if name_low == "q":
        return 3.0
    return 0.0


def score_answer_col(name: str) -> float:
    """Assign a score to a column name based on its likelihood of being answer content."""
    name_low = name.lower()
    if name_low in ("answertext", "answer_text", "answerbody", "answer_body", "response_text", "responsetext"):
        return 10.0
    if name_low == "answer":
        return 9.0
    if name_low == "response":
        return 8.0
    if "answer" in name_low:
        if any(x in name_low for x in ("id", "type", "time", "date", "author", "user", "score", "status")):
            return 2.0
        return 5.0
    if "response" in name_low:
        if any(x in name_low for x in ("id", "type", "time", "date", "author", "user", "score", "status")):
            return 1.0
        return 4.0
    if name_low == "a":
        return 3.0
    return 0.0


def find_qna_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """Identify the question and answer columns from the CSV schema."""
    question_col = None
    answer_col = None
    
    # Sort columns by their scores
    q_candidates = sorted(
        [(col, score_question_col(col)) for col in columns if score_question_col(col) > 0],
        key=lambda x: x[1],
        reverse=True
    )
    a_candidates = sorted(
        [(col, score_answer_col(col)) for col in columns if score_answer_col(col) > 0],
        key=lambda x: x[1],
        reverse=True
    )
    
    if q_candidates:
        question_col = q_candidates[0][0]
    if a_candidates:
        answer_col = a_candidates[0][0]
        
    return question_col, answer_col


def find_category_product_cols(columns: list[str]) -> tuple[str | None, str | None]:
    """Identify potential category and product information columns."""
    category_col = None
    product_col = None
    
    cat_exact = ["category", "categories", "group", "genre", "subcategory"]
    for col in columns:
        if col.lower() in cat_exact:
            category_col = col
            break
    if not category_col:
        for col in columns:
            if "category" in col.lower() or "group" in col.lower():
                if "id" not in col.lower():
                    category_col = col
                    break
                    
    prod_exact = ["asin", "product", "product_id", "productid", "item", "item_id", "itemid"]
    for col in columns:
        if col.lower() in prod_exact:
            product_col = col
            break
    if not product_col:
        for col in columns:
            if "product" in col.lower() or "item" in col.lower():
                if "id" not in col.lower() or col.lower() == "product_id":
                    product_col = col
                    break
                    
    return category_col, product_col


def stratified_sampling(df: pd.DataFrame, stratify_col: str, target_size: int, seed: int = 42) -> pd.DataFrame:
    """
    Perform robust proportional stratified sampling.
    Ensures that categories are represented proportionally and that the output length is exactly target_size.
    """
    if len(df) <= target_size:
        return df

    # Replace missing values in the stratification column to ensure they are grouped together
    df_strat = df.copy()
    df_strat[stratify_col] = df_strat[stratify_col].fillna("Unknown").astype(str)

    class_counts = df_strat[stratify_col].value_counts()
    total_rows = len(df_strat)
    
    # Calculate target counts for each class
    allocated_counts = {}
    for val, count in class_counts.items():
        allocated = int(round(target_size * count / total_rows))
        # Ensure at least 1 representative if the category contains rows
        allocated = max(1, min(count, allocated))
        allocated_counts[val] = allocated

    # Adjust rounding discrepancies to match target_size exactly
    current_sum = sum(allocated_counts.values())
    if current_sum > target_size:
        # Over-allocated: decrement categories with the largest allocations, keeping at least 1 sample
        sorted_keys = sorted(allocated_counts.keys(), key=lambda k: allocated_counts[k], reverse=True)
        diff = current_sum - target_size
        for k in sorted_keys:
            if diff == 0:
                break
            if allocated_counts[k] > 1:
                reduce_by = min(diff, allocated_counts[k] - 1)
                allocated_counts[k] -= reduce_by
                diff -= reduce_by
    elif current_sum < target_size:
        # Under-allocated: increment categories with remaining items
        sorted_keys = sorted(
            allocated_counts.keys(), 
            key=lambda k: (class_counts[k] - allocated_counts[k]), 
            reverse=True
        )
        diff = target_size - current_sum
        for k in sorted_keys:
            if diff == 0:
                break
            max_add = class_counts[k] - allocated_counts[k]
            if max_add > 0:
                add_val = min(diff, max_add)
                allocated_counts[k] += add_val
                diff -= add_val

    # Draw samples for each class
    sampled_dfs = []
    for val, n_samples in allocated_counts.items():
        group_df = df_strat[df_strat[stratify_col] == val]
        sampled_dfs.append(group_df.sample(n=n_samples, random_state=seed))

    sampled_df = pd.concat(sampled_dfs)
    # Shuffle the sampled dataset
    sampled_df = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled_df


def extract_qna_from_json(data: typing.Any) -> list[tuple[str, str]]:
    """
    Recursively crawl a JSON structure to dynamically extract question-answer pairs.
    Handles nested structures, lists of dicts, and direct question-to-answer dictionary mapping.
    """
    pairs = []
    
    def score_key(k: str, kind: str) -> float:
        k_low = k.lower()
        if kind == "q":
            if k_low == "question": return 5.0
            if k_low == "q": return 4.0
            if "question" in k_low: return 3.0
            if k_low == "query": return 2.0
            if "query" in k_low: return 1.0
        elif kind == "a":
            if k_low == "answer": return 5.0
            if k_low == "a": return 4.0
            if "answer" in k_low: return 3.0
            if k_low == "response": return 2.0
            if "response" in k_low: return 1.0
        return 0.0

    def traverse(node):
        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    # Check for question and answer keys in the dictionary
                    q_cols = sorted(
                        [k for k in item.keys() if isinstance(k, str) and score_key(k, "q") > 0],
                        key=lambda k: score_key(k, "q"),
                        reverse=True
                    )
                    a_cols = sorted(
                        [k for k in item.keys() if isinstance(k, str) and score_key(k, "a") > 0],
                        key=lambda k: score_key(k, "a"),
                        reverse=True
                    )
                    if q_cols and a_cols:
                        q_val = item[q_cols[0]]
                        a_val = item[a_cols[0]]
                        if isinstance(q_val, str) and isinstance(a_val, str):
                            pairs.append((q_val.strip(), a_val.strip()))
                    else:
                        traverse(item)
                else:
                    traverse(item)
        elif isinstance(node, dict):
            # Check if this is a flat QA mapping dictionary
            is_flat_qa = True
            has_question_like_key = False
            question_starters = ("how", "what", "why", "can", "do", "is", "are", "where", "when", "who", "which")
            
            if len(node) == 0:
                is_flat_qa = False
            else:
                for k, v in node.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        is_flat_qa = False
                        break
                    k_stripped = k.strip()
                    if k_stripped.endswith("?") or any(k_stripped.lower().startswith(w) for w in question_starters):
                        has_question_like_key = True
            
            if is_flat_qa and has_question_like_key:
                for k, v in node.items():
                    pairs.append((k.strip(), v.strip()))
            else:
                for v in node.values():
                    traverse(v)

    traverse(data)
    return pairs


def load_amazon_qa(file_path: str = "data/single_qna.csv", sample_size: int = 15000) -> list[Document]:
    """
    Loads Amazon QA data from a CSV file, reduces it if necessary, and converts it to LangChain Documents.
    """
    logger.info(f"Starting to load Amazon QA from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Amazon QA CSV file not found at: {file_path}")
        
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {file_path}")
        raise ValueError(f"Invalid CSV structure or read error: {e}") from e

    columns = list(df.columns)
    q_col, a_col = find_qna_columns(columns)
    
    # Exception handling for missing QA columns
    if not q_col or not a_col:
        raise ValueError(
            f"Could not automatically identify question and answer columns in {file_path}. "
            f"Columns: {columns}"
        )
        
    cat_col, prod_col = find_category_product_cols(columns)
    
    print("--- Amazon QA Dataset Schema Inspection ---")
    print(f"Detected Question Column: '{q_col}'")
    print(f"Detected Answer Column:   '{a_col}'")
    print(f"Detected Category Column: '{cat_col}'")
    print(f"Detected Product Column:  '{prod_col}'")

    original_size = len(df)
    logger.info(f"Loaded {original_size} rows from {file_path}")

    # Sampling strategy selection and execution
    if original_size > sample_size:
        # Prioritize stratified sampling if category or product column exists
        stratify_target = cat_col or prod_col
        if stratify_target:
            sampling_strategy = "Stratified sampling"
            print(f"Sampling Strategy:        {sampling_strategy} (on column '{stratify_target}')")
            df_sampled = stratified_sampling(df, stratify_target, sample_size, seed=42)
        else:
            sampling_strategy = "Random sampling"
            print(f"Sampling Strategy:        {sampling_strategy}")
            df_sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        sampling_strategy = "None (Dataset is smaller than sample threshold)"
        print(f"Sampling Strategy:        {sampling_strategy}")
        df_sampled = df

    final_size = len(df_sampled)
    print(f"Final Amazon QA Size:     {final_size} rows")
    print("-------------------------------------------")

    documents = []
    source_filename = os.path.basename(file_path)
    
    for idx, row in df_sampled.iterrows():
        question_text = str(row[q_col]).strip()
        answer_text = str(row[a_col]).strip()
        
        page_content = f"Question: {question_text}\nAnswer: {answer_text}"
        
        metadata = {
            "source_type": "amazon_qa",
            "source_file": source_filename,
            "question": question_text,
            "answer": answer_text
        }
        
        if cat_col and pd.notna(row[cat_col]):
            metadata["category"] = str(row[cat_col])
        if prod_col and pd.notna(row[prod_col]):
            metadata["product"] = str(row[prod_col])
            
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"Created {len(documents)} Amazon QA LangChain Document objects")
    return documents


def load_ecommerce_faq(file_path: str = "data/Ecommerce_FAQ_Chatbot_dataset.json") -> list[Document]:
    """
    Loads Ecommerce FAQ data from a JSON file and converts it to LangChain Documents.
    """
    logger.info(f"Starting to load Ecommerce FAQ from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ecommerce FAQ JSON file not found at: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from: {file_path}")
        raise ValueError(f"Invalid JSON file format: {e}") from e
    except Exception as e:
        logger.error(f"Failed to read file: {file_path}")
        raise

    # Dynamically extract question-answer pairs
    qa_pairs = extract_qna_from_json(data)
    
    if not qa_pairs:
        raise ValueError(
            f"Could not automatically detect any FAQ question-answer pairs in the JSON structure of {file_path}."
        )

    logger.info(f"Extracted {len(qa_pairs)} FAQ pairs from {file_path}")
    
    documents = []
    source_filename = os.path.basename(file_path)
    
    for question, answer in qa_pairs:
        page_content = f"Question: {question}\nAnswer: {answer}"
        metadata = {
            "source_type": "ecommerce_faq",
            "source_file": source_filename,
            "question": question,
            "answer": answer
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(f"Created {len(documents)} Ecommerce FAQ LangChain Document objects")
    return documents


def load_manuals(manuals_dir: str = "data/manuals/") -> list[Document]:
    """
    Loads all PDFs in manuals_dir, parses them page-by-page, and converts to LangChain Documents.
    """
    logger.info(f"Starting to load manual PDFs from: {manuals_dir}")
    if not os.path.exists(manuals_dir):
        raise FileNotFoundError(f"Manuals directory not found at: {manuals_dir}")
        
    pdf_files = [f for f in os.listdir(manuals_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {manuals_dir}")
        return []

    documents = []
    
    for filename in pdf_files:
        pdf_path = os.path.join(manuals_dir, filename)
        logger.info(f"Loading PDF file: {pdf_path}")
        
        try:
            reader = pypdf.PdfReader(pdf_path)
            num_pages = len(reader.pages)
            
            # Exception handling for empty PDFs (0 pages)
            if num_pages == 0:
                raise ValueError(f"PDF manual '{filename}' is empty (has 0 pages).")
                
            pdf_text_extracted = False
            
            for page_idx in range(num_pages):
                page = reader.pages[page_idx]
                page_text = page.extract_text()
                
                # Check for None and strip
                page_text_clean = (page_text or "").strip()
                
                if page_text_clean:
                    pdf_text_extracted = True
                    
                page_num_1based = page_idx + 1
                
                metadata = {
                    "source_type": "manual",
                    "source_file": filename,
                    "page": page_num_1based,
                    "page_number": page_num_1based
                }
                
                documents.append(Document(page_content=page_text_clean, metadata=metadata))
                
            # Exception handling if PDF has pages but contains absolutely no extractable text
            if not pdf_text_extracted:
                raise ValueError(
                    f"PDF manual '{filename}' contains no extractable text. "
                    f"The document may be scanned or empty."
                )
                
            logger.info(f"Loaded {num_pages} pages from manual: {filename}")
            
        except Exception as e:
            logger.error(f"Error loading PDF manual '{filename}': {e}")
            raise

    logger.info(f"Created {len(documents)} PDF page LangChain Document objects")
    return documents


def load_all_documents(data_dir: str = "data") -> list[Document]:
    """
    Loads documents from all sources and returns the combined list.
    """
    logger.info(f"Loading all data sources in directory: {data_dir}")
    
    # Paths for sources
    amazon_csv = os.path.join(data_dir, "single_qna.csv")
    faq_json = os.path.join(data_dir, "Ecommerce_FAQ_Chatbot_dataset.json")
    manuals_path = os.path.join(data_dir, "manuals")
    
    # Load each source
    amazon_docs = load_amazon_qa(amazon_csv)
    faq_docs = load_ecommerce_faq(faq_json)
    manual_docs = load_manuals(manuals_path)
    
    # Combine documents
    all_docs = amazon_docs + faq_docs + manual_docs
    
    # Print the required summary counts
    print("\n================ DATA LOADING SUMMARY ================")
    print(f"Number of Amazon QA documents loaded:    {len(amazon_docs):,}")
    print(f"Number of Ecommerce FAQ documents loaded: {len(faq_docs):,}")
    print(f"Number of PDF documents loaded:          {len(manual_docs):,}")
    print(f"Total documents loaded:                  {len(all_docs):,}")
    print("======================================================\n")
    
    return all_docs


if __name__ == "__main__":
    print("Executing data loader module directly...\n")
    try:
        documents = load_all_documents()
        
        # Display samples from each source type
        amazon_samples = [d for d in documents if d.metadata["source_type"] == "amazon_qa"]
        faq_samples = [d for d in documents if d.metadata["source_type"] == "ecommerce_faq"]
        manual_samples = [d for d in documents if d.metadata["source_type"] == "manual"]
        
        print("------------------------------------------------------")
        print("SAMPLES OF LOADED DOCUMENTS FROM EACH SOURCE")
        print("------------------------------------------------------")
        
        if amazon_samples:
            print("\n[SAMPLE 1] Amazon QA Document:")
            sample = amazon_samples[0]
            print(f"Metadata:     {sample.metadata}")
            # Print first 200 characters of content
            print(f"Page Content:\n{sample.page_content[:400]}...")
            
        if faq_samples:
            print("\n[SAMPLE 2] Ecommerce FAQ Document:")
            sample = faq_samples[0]
            print(f"Metadata:     {sample.metadata}")
            print(f"Page Content:\n{sample.page_content[:400]}...")
            
        if manual_samples:
            print("\n[SAMPLE 3] PDF Manual Document:")
            sample = manual_samples[0]
            print(f"Metadata:     {sample.metadata}")
            print(f"Page Content:\n{sample.page_content[:400]}...")
            
    except Exception as ex:
        logger.exception("An error occurred during data loading execution")
