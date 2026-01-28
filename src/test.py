from src.state import *
from src.prompt import *
from src.llmconfig import *
from langgraph.graph import StateGraph,START,END
from src.agents.clerk import *
from src.agents.judge import *
from src.agents.denfendant import *
from src.agents.prosecutor import *



graph = StateGraph(CourtState)
graph.add_node("judge_check",judge_check)
graph.add_node("clerk_rules",clerk_rules)
graph.add_node("judge_open",judge_open)
graph.add_node("right_notify",right_notify)
graph.add_node("pros_indictment",pros_indictment)
graph.add_node("defense_defense_object_control",defense_object_control) # 路由节点
graph.add_node("defense_objection",defense_objection)
graph.add_node("pros_question",pros_question)
graph.add_node("defense_reply",defense_reply)
graph.add_node("defense_question_control",defense_question_control) # 路由节点
graph.add_node("defense_question",defense_question)
graph.add_node("pros_summary",pros_summary)
graph.add_node("defense_summary",defense_summary)
graph.add_node("judge_start_evidence",judge_start_evidence)
graph.add_node("pros_evidence_decision", pros_evidence_decision) # 路由节点
graph.add_node("pros_show_evidence",pros_show_evidence)
graph.add_node("defense_cross",defense_cross)
graph.add_node("judge_confirm",judge_confirm)
graph.add_node("defense_evidence_control",defense_evidence_control) # 路由节点
graph.add_node("defense_show_evidence",defense_show_evidence)
graph.add_node("pros_cross",pros_cross)
graph.add_node("judge_start_debate",judge_start_debate)
graph.add_node("pros_statement",pros_statement)
graph.add_node("defense_self_statement",defense_self_statement)
graph.add_node("defense_statement",defense_statement)
graph.add_node("judge_summary",judge_summary)
graph.add_node("focus",focus) # 路由节点
graph.add_node("pros_focus",pros_focus) # 路由节点
graph.add_node("defense_focus",defense_focus)
graph.add_node("pros_sumup",pros_sumup)
graph.add_node("defense_sumup",defense_sumup)
graph.add_node("defense_final_statement",defense_final_statement)
graph.add_node("judge_verdict",judge_verdict)


graph.add_edge(START,"clerk_rules")
graph.add_edge("clerk_rules","judge_open")
graph.add_edge("judge_open","judge_check")
graph.add_edge("judge_check","right_notify")
graph.add_edge("right_notify","pros_indictment")
graph.add_edge("pros_indictment","defense_defense_object_control")  #之后图的跳转由Command在控制
graph.add_edge("defense_objection","pros_question")
graph.add_conditional_edges("pros_question",pros_round_control,
                            {
                                "defense_reply":"defense_reply",
                                "defense_question":"defense_question"
                            })
graph.add_conditional_edges("defense_reply",reply_control,
                            {
                                "defense_question": "defense_question_control", # 跳转到路由节点，之后由Command控制
                                "pros_question": "pros_question"
                            })
graph.add_edge("defense_question","defense_reply")
graph.add_edge("pros_summary","defense_summary")
graph.add_edge("defense_summary", "judge_start_evidence")
graph.add_edge("judge_start_evidence","pros_evidence_decision") # 连接到路由节点，之后由Command控制
graph.add_edge("pros_show_evidence","defense_cross")
graph.add_edge("defense_cross","pros_evidence_decision")
graph.add_edge("judge_confirm","defense_evidence_control") # 连接到路由节点，之后由Command控制
graph.add_edge("defense_show_evidence","pros_cross")
graph.add_edge("pros_cross","defense_evidence_control")
graph.add_edge("judge_start_debate","pros_statement")
graph.add_edge("pros_statement","defense_self_statement")
graph.add_edge("defense_self_statement","defense_statement")
graph.add_edge("defense_statement","judge_summary")
graph.add_edge("judge_summary","focus") # 连接到路由节点，之后由Command控制
graph.add_edge("defense_focus","pros_focus") # 连接到路由节点，之后由Command控制
graph.add_edge("pros_sumup","defense_sumup")
graph.add_edge("defense_sumup","defense_final_statement")
graph.add_edge("defense_final_statement","judge_verdict")
graph.add_edge("judge_verdict",END)


app = graph.compile()
