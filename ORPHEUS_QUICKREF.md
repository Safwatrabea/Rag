# Quick Reference: Custom Orpheus Model

## 🎯 **What Changed**

**Tab 3 (General Helper)** now uses: `canopylabs/orpheus-arabic-saudi`

**This is a custom Saudi-optimized model available in your Groq account.**

## 🚀 **Current Model Setup**

```
┌─────────────┬──────────────────────────────────┬─────────────┐
│    Tab      │            Model                 │   Purpose   │
├─────────────┼──────────────────────────────────┼─────────────┤
│  Tab 1      │  llama-3.3-70b-versatile        │  RAG Chat   │
│  Tab 2      │  llama-3.3-70b-versatile        │  Reports    │
│  Tab 3      │  canopylabs/orpheus-arabic-saudi │  General    │
└─────────────┴──────────────────────────────────┴─────────────┘
```

## 🔧 **Code Location**

**File**: `rag.py`  
**Function**: `_setup_groq_llm()`  
**Lines**: ~300-325

```python
self.llm_groq = ChatGroq(
    model_name="canopylabs/orpheus-arabic-saudi",  # ← Custom model
    temperature=0.7,
    streaming=True
)
```

## 🎛️ **Startup Check**

Look for this message when starting the app:

```
🌙 Attempting to initialize Orpheus Arabic Saudi (Custom Saudi-native model)...
✅ Groq General Chat LLM initialized (Orpheus Arabic Saudi - Custom)
```

## ⚠️ **If Orpheus Fails**

Fallback sequence:
1. **Orpheus** → Try first
2. **Llama 3.1 8B** → If Orpheus fails
3. **GPT-4omini** → If Groq down

## ✅ **Quick Test**

```bash
# Start app
streamlit run app.py

# Go to Tab 3
# Ask: "ايش وضع السوق اليوم؟"
# Expected: Natural Saudi-style response
```

## 📝 **Requirements**

- `GROQ_API_KEY` must be set in `.env`
- Your Groq account must have access to `canopylabs/orpheus-arabic-saudi`
- `langchain-groq` must be installed

## 🎊 **Result**

Tab 3 now speaks with authentic Saudi dialect and cultural understanding! 🇸🇦
