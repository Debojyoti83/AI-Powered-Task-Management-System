import pandas as pd
import numpy as np
import nltk
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

print("="*60)
print("TASK MANAGEMENT SYSTEM - NLP PREPROCESSING")
print("="*60)

class TaskDescriptionNLPProcessor:
    """
    Comprehensive NLP preprocessing pipeline for task descriptions
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stop words specific to task management
        custom_stopwords = {
            'task', 'project', 'work', 'need', 'required', 'please', 
            'complete', 'finish', 'done', 'update', 'review', 'check',
            'ensure', 'make', 'sure', 'would', 'could', 'should',
            'team', 'user', 'system', 'process', 'implement', 'create'
        }
        self.stop_words.update(custom_stopwords)
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}')
        self.number_pattern = re.compile(r'\b\d+\b')
        
    def clean_text(self, text):
        """
        Basic text cleaning
        """
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' ', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub(' ', text)
        
        # Remove excessive numbers (keep version numbers like 2.0, 3.5)
        text = re.sub(r'\b\d{4,}\b', ' ', text)  # Remove long numbers
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text):
        """
        Tokenize text into words
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter out tokens that are too short or just punctuation
        tokens = [token for token in tokens if len(token) > 2 and not token in string.punctuation]
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from tokens
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """
        Apply lemmatization to tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_description(self, text, include_stemming=True, include_lemmatization=True):
        """
        Complete preprocessing pipeline for a single description
        """
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Step 3: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 4: Apply stemming (optional)
        if include_stemming:
            stemmed_tokens = self.stem_tokens(tokens)
        else:
            stemmed_tokens = tokens
            
        # Step 5: Apply lemmatization (optional)
        if include_lemmatization:
            lemmatized_tokens = self.lemmatize_tokens(tokens)
        else:
            lemmatized_tokens = tokens
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'stemmed_tokens': stemmed_tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'processed_text': ' '.join(lemmatized_tokens)
        }

def analyze_text_statistics(df, text_column):
    """
    Analyze text statistics
    """
    print(f"\n1. TEXT STATISTICS ANALYSIS")
    print("="*40)
    
    # Calculate text statistics
    df['text_length'] = df[text_column].astype(str).str.len()
    df['word_count'] = df[text_column].astype(str).str.split().str.len()
    df['sentence_count'] = df[text_column].astype(str).apply(lambda x: len(sent_tokenize(x)) if x else 0)
    
    print(f"Text Length Statistics:")
    print(f"  Mean: {df['text_length'].mean():.2f} characters")
    print(f"  Median: {df['text_length'].median():.2f} characters")
    print(f"  Max: {df['text_length'].max()} characters")
    print(f"  Min: {df['text_length'].min()} characters")
    
    print(f"\nWord Count Statistics:")
    print(f"  Mean: {df['word_count'].mean():.2f} words")
    print(f"  Median: {df['word_count'].median():.2f} words")
    print(f"  Max: {df['word_count'].max()} words")
    print(f"  Min: {df['word_count'].min()} words")
    
    print(f"\nSentence Count Statistics:")
    print(f"  Mean: {df['sentence_count'].mean():.2f} sentences")
    print(f"  Median: {df['sentence_count'].median():.2f} sentences")
    print(f"  Max: {df['sentence_count'].max()} sentences")
    print(f"  Min: {df['sentence_count'].min()} sentences")
    
    return df

def extract_keywords_by_category(df, processed_column, category_column):
    """
    Extract top keywords for each category
    """
    print(f"\n2. KEYWORD EXTRACTION BY CATEGORY")
    print("="*40)
    
    category_keywords = {}
    
    for category in df[category_column].unique():
        if pd.isna(category):
            continue
            
        category_texts = df[df[category_column] == category][processed_column]
        
        # Combine all texts for this category
        all_words = []
        for text in category_texts:
            if isinstance(text, str) and text:
                all_words.extend(text.split())
        
        # Count word frequencies
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(10)
        
        category_keywords[category] = top_words
        
        print(f"\n{category}:")
        for word, freq in top_words[:7]:  # Show top 7
            print(f"  {word}: {freq}")
    
    return category_keywords

def create_visualizations(df, text_stats_cols):
    """
    Create visualizations for text analysis
    """
    print(f"\n3. CREATING VISUALIZATIONS")
    print("="*40)
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Task Description Text Analysis', fontsize=16, fontweight='bold')
    
    # 1. Text length distribution
    axes[0,0].hist(df['text_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Text Length')
    axes[0,0].set_xlabel('Characters')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Word count distribution
    axes[0,1].hist(df['word_count'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_title('Distribution of Word Count')
    axes[0,1].set_xlabel('Words')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Text length by category
    categories = df['category'].value_counts().head(8).index
    category_data = [df[df['category'] == cat]['text_length'].values for cat in categories]
    axes[1,0].boxplot(category_data, labels=[cat[:8] + '...' if len(cat) > 8 else cat for cat in categories])
    axes[1,0].set_title('Text Length by Category')
    axes[1,0].set_ylabel('Characters')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Word count by priority
    priority_order = ['Low', 'Medium', 'High', 'Critical']
    priority_data = [df[df['priority'] == pri]['word_count'].values for pri in priority_order if pri in df['priority'].values]
    priority_labels = [pri for pri in priority_order if pri in df['priority'].values]
    axes[1,1].boxplot(priority_data, labels=priority_labels)
    axes[1,1].set_title('Word Count by Priority')
    axes[1,1].set_ylabel('Words')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Visualizations created successfully!")

def main():
    """
    Main function to run the NLP preprocessing pipeline
    """
    print("Loading cleaned dataset...")
    
    # Load the cleaned dataset
    try:
        df = pd.read_csv('task_management_dataset_cleaned.csv')
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: 'task_management_dataset_cleaned.csv' not found!")
        print("Please make sure the cleaned dataset file is in the current directory.")
        return
    
    # Initialize the NLP processor
    processor = TaskDescriptionNLPProcessor()
    
    # Analyze original text statistics
    print(f"\nOriginal descriptions sample:")
    print(df['description'].head(3).tolist())
    
    df = analyze_text_statistics(df, 'description')
    
    # Apply NLP preprocessing
    print(f"\n4. APPLYING NLP PREPROCESSING")
    print("="*40)
    print("Processing task descriptions...")
    
    # Process descriptions
    processed_results = []
    for idx, description in enumerate(df['description']):
        if idx % 10000 == 0:
            print(f"  Processed {idx:,} descriptions...")
        
        result = processor.preprocess_description(description)
        processed_results.append(result)
    
    # Add processed columns to dataframe
    df['cleaned_text'] = [r['cleaned_text'] for r in processed_results]
    df['tokens'] = [r['tokens'] for r in processed_results]
    df['stemmed_tokens'] = [r['stemmed_tokens'] for r in processed_results]
    df['lemmatized_tokens'] = [r['lemmatized_tokens'] for r in processed_results]
    df['processed_text'] = [r['processed_text'] for r in processed_results]
    
    print(f"✅ Processed {len(df):,} task descriptions successfully!")
    
    # Show examples of preprocessing
    print(f"\n5. PREPROCESSING EXAMPLES")
    print("="*40)
    
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(f"Original: {df.iloc[i]['description'][:100]}...")
        print(f"Cleaned:  {df.iloc[i]['cleaned_text'][:100]}...")
        print(f"Tokens:   {df.iloc[i]['tokens'][:10]}")
        print(f"Stemmed:  {df.iloc[i]['stemmed_tokens'][:10]}")
        print(f"Lemmatized: {df.iloc[i]['lemmatized_tokens'][:10]}")
    
    # Extract keywords by category
    category_keywords = extract_keywords_by_category(df, 'processed_text', 'category')
    
    # Calculate processing statistics
    print(f"\n6. PROCESSING STATISTICS")
    print("="*40)
    
    # Token statistics
    df['token_count'] = df['tokens'].apply(len)
    df['processed_token_count'] = df['processed_text'].str.split().str.len()
    
    print(f"Token Count Statistics (after preprocessing):")
    print(f"  Mean: {df['processed_token_count'].mean():.2f} tokens")
    print(f"  Median: {df['processed_token_count'].median():.2f} tokens")
    print(f"  Max: {df['processed_token_count'].max()} tokens")
    print(f"  Min: {df['processed_token_count'].min()} tokens")
    
    # Vocabulary size
    all_tokens = []
    for tokens in df['lemmatized_tokens']:
        all_tokens.extend(tokens)
    
    vocabulary = set(all_tokens)
    print(f"\nVocabulary Statistics:")
    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Unique tokens (vocabulary size): {len(vocabulary):,}")
    print(f"  Average tokens per description: {len(all_tokens)/len(df):.2f}")
    
    # Most common words overall
    word_freq = Counter(all_tokens)
    print(f"\nTop 15 most common words:")
    for word, freq in word_freq.most_common(15):
        print(f"  {word}: {freq:,}")
    
    # Create visualizations
    create_visualizations(df, ['text_length', 'word_count', 'sentence_count'])
    
    # Save processed dataset
    print(f"\n7. SAVING PROCESSED DATASET")
    print("="*40)
    
    # Select columns to save
    columns_to_save = [
        'task_id', 'title', 'description', 'category', 'priority', 'status',
        'created_date', 'due_date', 'completion_date', 'assigned_to', 'created_by',
        'manager', 'department', 'estimated_hours', 'actual_hours', 'complexity_score',
        'dependencies_count', 'user_current_workload', 'user_experience_level',
        'task_age_days', 'is_overdue', 'task_duration_days', 'effort_variance_ratio',
        'priority_score', 'days_to_due', 'workload_intensity',
        'cleaned_text', 'processed_text', 'text_length', 'word_count', 
        'sentence_count', 'token_count', 'processed_token_count'
    ]
    
    df_to_save = df[columns_to_save].copy()
    df_to_save.to_csv('task_management_dataset_nlp_processed.csv', index=False)
    
    print(f"✅ Processed dataset saved as 'task_management_dataset_nlp_processed.csv'")
    print(f"   Shape: {df_to_save.shape}")
    print(f"   New NLP columns added: 7")
    
    # Final summary
    print(f"\n8. NLP PREPROCESSING SUMMARY")
    print("="*40)
    print(f"✅ TEXT CLEANING:")
    print(f"   • Removed URLs, emails, phone numbers")
    print(f"   • Normalized text case and whitespace")
    print(f"   • Removed excessive punctuation and numbers")
    
    print(f"\n✅ TOKENIZATION:")
    print(f"   • Tokenized {len(df):,} descriptions")
    print(f"   • Generated {len(all_tokens):,} total tokens")
    print(f"   • Created vocabulary of {len(vocabulary):,} unique words")
    
    print(f"\n✅ STOPWORD REMOVAL:")
    print(f"   • Removed common English stopwords")
    print(f"   • Removed task-specific stopwords")
    print(f"   • Filtered short tokens and punctuation")
    
    print(f"\n✅ STEMMING & LEMMATIZATION:")
    print(f"   • Applied Porter stemming")
    print(f"   • Applied WordNet lemmatization")
    print(f"   • Normalized word forms")
    
    return df_to_save

if __name__ == "__main__":
    processed_df = main()