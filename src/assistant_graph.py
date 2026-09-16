from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# =========================================================
# 1. Workflow state
# =========================================================

class AssistantState(TypedDict):
    prediction: str
    confidence: float
    explanation: str


# =========================================================
# 2. Decision function
# =========================================================

def route_prediction(state: AssistantState):
    prediction = state["prediction"]

    if prediction == "Normal":
        return "normal"

    elif prediction == "Ischemia":
        return "ischemia"

    elif prediction == "Bleeding":
        return "bleeding"

    else:
        raise ValueError(f"Unknown prediction: {prediction}")


# =========================================================
# 3. Explanation nodes
# =========================================================

def normal_explanation(state: AssistantState):
    return {
        "explanation": (
            f"The model classified this CT image as Normal "
            f"with {state['confidence']:.2f}% confidence. "
            "No abnormality was detected by the model. "
            "This result should still be reviewed by a qualified "
            "medical professional."
        )
    }


def ischemia_explanation(state: AssistantState):
    return {
        "explanation": (
            f"The model classified this CT image as Ischemia "
            f"with {state['confidence']:.2f}% confidence. "
            "Ischemia may indicate reduced blood flow to brain tissue. "
            "Urgent medical assessment is recommended."
        )
    }


def bleeding_explanation(state: AssistantState):
    return {
        "explanation": (
            f"The model classified this CT image as Bleeding "
            f"with {state['confidence']:.2f}% confidence. "
            "Possible brain bleeding is a medical emergency. "
            "Immediate professional medical assessment is required."
        )
    }


# =========================================================
# 4. Build LangGraph
# =========================================================

graph_builder = StateGraph(AssistantState)


# Add explanation nodes
graph_builder.add_node(
    "normal_explanation",
    normal_explanation
)

graph_builder.add_node(
    "ischemia_explanation",
    ischemia_explanation
)

graph_builder.add_node(
    "bleeding_explanation",
    bleeding_explanation
)


# Conditional routing
graph_builder.add_conditional_edges(
    START,
    route_prediction,
    {
        "normal": "normal_explanation",
        "ischemia": "ischemia_explanation",
        "bleeding": "bleeding_explanation",
    }
)


# Connect nodes to END
graph_builder.add_edge(
    "normal_explanation",
    END
)

graph_builder.add_edge(
    "ischemia_explanation",
    END
)

graph_builder.add_edge(
    "bleeding_explanation",
    END
)


# Compile graph
assistant_graph = graph_builder.compile()


# =========================================================
# 5. Test the workflow
# =========================================================

if __name__ == "__main__":

    test_cases = [
        {
            "prediction": "Normal",
            "confidence": 94.15,
            "explanation": "",
        },
        {
            "prediction": "Ischemia",
            "confidence": 88.50,
            "explanation": "",
        },
        {
            "prediction": "Bleeding",
            "confidence": 96.20,
            "explanation": "",
        },
    ]

    for test_case in test_cases:

        result = assistant_graph.invoke(test_case)

        print("\n==============================")
        print(f"Prediction: {test_case['prediction']}")
        print(f"Confidence: {test_case['confidence']:.2f}%")
        print("Assistant Response:")
        print(result["explanation"])