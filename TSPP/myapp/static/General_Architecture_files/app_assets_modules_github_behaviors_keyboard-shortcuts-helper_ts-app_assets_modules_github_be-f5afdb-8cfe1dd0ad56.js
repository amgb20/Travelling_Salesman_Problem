"use strict";(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["app_assets_modules_github_behaviors_keyboard-shortcuts-helper_ts-app_assets_modules_github_be-f5afdb"],{94313:(e,t,n)=>{n.d(t,{Ty:()=>r,YE:()=>l,Zf:()=>i});var s=n(11793);let r=()=>{let e=document.querySelector("meta[name=keyboard-shortcuts-preference]");return!e||"all"===e.content},l=e=>/Enter|Arrow|Escape|Meta|Control|Esc/.test(e)||e.includes("Alt")&&e.includes("Shift"),i=e=>{let t=(0,s.EL)(e);return!!r()||l(t)}},48804:(e,t,n)=>{n.d(t,{L$:()=>u,SE:()=>j,nj:()=>d});var s,r=n(56959),l=n(59753),i=n(40987),o=n(36071),a=n(65935),c=n(58700);function u(e){if(e.querySelector(".js-task-list-field")){let t=e.querySelectorAll("task-lists");for(let e of t)if(e instanceof i.Z){e.disabled=!1;let t=e.querySelectorAll("button");for(let e of t)e.disabled=!1}}}function d(e){for(let t of e.querySelectorAll("task-lists"))if(t instanceof i.Z){t.disabled=!0;let e=t.querySelectorAll("button");for(let t of e)t.disabled=!0}}function f(e,t,n){let s=e.querySelector(".js-comment-update");d(e),R(e);let r=s.elements.namedItem("task_list_track");r instanceof Element&&r.remove();let l=s.elements.namedItem("task_list_operation");l instanceof Element&&l.remove();let i=document.createElement("input");i.setAttribute("type","hidden"),i.setAttribute("name","task_list_track"),i.setAttribute("value",t),s.appendChild(i);let o=document.createElement("input");if(o.setAttribute("type","hidden"),o.setAttribute("name","task_list_operation"),o.setAttribute("value",JSON.stringify(n)),s.appendChild(o),!s.elements.namedItem("task_list_key")){let e=s.querySelector(".js-task-list-field"),t=e.getAttribute("name"),n=t.split("[")[0],r=document.createElement("input");r.setAttribute("type","hidden"),r.setAttribute("name","task_list_key"),r.setAttribute("value",n),s.appendChild(r)}e.classList.remove("is-comment-stale"),(0,c.Bt)(s)}(0,o.N7)(".js-task-list-container .js-task-list-field",function(e){let t=e.closest(".js-task-list-container");u(t),R(t)}),(0,o.N7)(".js-convert-tasklist-to-block-enabled .contains-task-list",function(e){let t=$(e);if(!t)return;let n=Array.from(t.children).some(e=>e.classList.contains("task-list-item-convert-container"));if(n)return;let s=e.ownerDocument.querySelector(".js-convert-to-block-template"),r=s?.content.cloneNode(!0);r&&t.appendChild(r)}),(0,l.on)("task-lists-move","task-lists",function(e){let{src:t,dst:n}=e.detail,s=e.currentTarget.closest(".js-task-list-container");f(s,"reordered",{operation:"move",src:t,dst:n})}),(0,l.on)("task-lists-check","task-lists",function(e){let{position:t,checked:n}=e.detail,s=e.currentTarget.closest(".js-task-list-container");f(s,`checked:${n?1:0}`,{operation:"check",position:t,checked:n})}),(0,l.on)("click",".js-convert-to-block-button",function(e){let t=$(e.target);if(!t)return;let n=t.closest("task-lists");if(!n)throw Error("parent not found");let s=j(t),r=e.currentTarget.closest(".js-task-list-container");f(r,"converted",{operation:"convert_to_block",position:s})}),(0,a.AC)(".js-task-list-container .js-comment-update",async function(e,t){let n;let s=e.closest(".js-task-list-container"),r=e.elements.namedItem("task_list_track");r instanceof Element&&r.remove();let l=e.elements.namedItem("task_list_operation");l instanceof Element&&l.remove();try{n=await t.json()}catch(t){let e;try{e=JSON.parse(t.response.text)}catch(e){}if(e&&e.stale){let e=s.querySelector(".js-task-list-field");e.classList.add("session-resumable-canceled"),e.classList.remove("js-session-resumable")}else 422===t.response.status||window.location.reload()}n&&(l&&n.json.source&&(s.querySelector(".js-task-list-field").value=n.json.source),u(s),requestAnimationFrame(()=>R(s)))});let m=!1,p=!1,h=null;function b(e){let t="insertLineBreak"===e.inputType;m=!!t}function y(e){if(!m){let t="insertLineBreak"===e.inputType;if(!t)return}let t=e.target;g(t),m=!1}function g(e){let t=A(e.value,[e.selectionStart,e.selectionEnd]);void 0!==t&&w(e,t)}function w(e,t){if(null===h||!0===h){e.contentEditable="true";try{let n;m=!1,t.commandId===s.insertText?(n=t.autocompletePrefix,null!==t.writeSelection[0]&&null!==t.writeSelection[1]&&(e.selectionStart=t.writeSelection[0],e.selectionEnd=t.writeSelection[1])):e.selectionStart=t.selection[0],h=document.execCommand(t.commandId,!1,n)}catch(e){h=!1}e.contentEditable="false"}if(!h){try{document.execCommand("ms-beginUndoUnit")}catch(e){}e.value=t.text;try{document.execCommand("ms-endUndoUnit")}catch(e){}e.dispatchEvent(new CustomEvent("input",{bubbles:!0,cancelable:!0}))}null!=t.selection[0]&&null!=t.selection[1]&&(e.selectionStart=t.selection[0],e.selectionEnd=t.selection[1])}function k(e){if(!p&&"Enter"===e.key&&e.shiftKey&&!e.metaKey){let t=e.target,n=L(t.value,[t.selectionStart,t.selectionEnd]);void 0!==n&&(w(t,n),e.preventDefault(),(0,l.f)(t,"change"))}}function v(){p=!0}function E(){p=!1}function S(e){if(p)return;if("Escape"===e.key){C(e);return}if("Tab"!==e.key)return;let t=e.target,n=_(t.value,[t.selectionStart,t.selectionEnd],e.shiftKey);void 0!==n&&(e.preventDefault(),w(t,n))}(0,o.N7)(".js-task-list-field",{subscribe:e=>(0,r.qC)((0,r.RB)(e,"keydown",S),(0,r.RB)(e,"keydown",k),(0,r.RB)(e,"beforeinput",b),(0,r.RB)(e,"input",y),(0,r.RB)(e,"compositionstart",v),(0,r.RB)(e,"compositionend",E))}),function(e){e.insertText="insertText",e.delete="delete"}(s||(s={}));let x=/^(\s*)?/;function L(e,t){let n=t[0];if(!n||!e)return;let r=e.substring(0,n).split("\n"),l=r[r.length-1],i=l?.match(x);if(!i)return;let o=i[1]||"",a=`
${o}`;return{text:e.substring(0,n)+a+e.substring(n),autocompletePrefix:a,selection:[n+a.length,n+a.length],commandId:s.insertText,writeSelection:[null,null]}}let T=/^(\s*)([*-]|(\d+)\.)\s(\[[\sx]\]\s)?/;function q(e,t){let n=e.split("\n");return(n=n.map(e=>{if(e.replace(/^\s+/,"").startsWith(`${t}.`)){let n=e.replace(`${t}`,`${t+1}`);return t+=1,n}return e})).join("\n")}function A(e,t){let n=t[0];if(!n||!e)return;let r=e.substring(0,n).split("\n"),l=r[r.length-2],i=l?.match(T);if(!i)return;let o=i[0],a=i[1],c=i[2],u=parseInt(i[3],10),d=Boolean(i[4]),f=!isNaN(u),m=f?`${u+1}.`:c,p=`${m} ${d?"[ ] ":""}`,h=e.indexOf("\n",n);h<0&&(h=e.length);let b=e.substring(n,h);b.startsWith(p)&&(p="");let y=l.replace(o,"").trim().length>0||b.trim().length>0;if(y){let t=`${a}${p}`,r=e.substring(n),l=t.length,i=[null,null],o=e.substring(0,n)+t+r;return f&&!e.substring(n).match(/^\s*$/g)&&(t+=r=q(e.substring(n),u+1),i=[n,n+t.length],o=e.substring(0,n)+t),{text:o,autocompletePrefix:t,selection:[n+l,n+l],commandId:s.insertText,writeSelection:i}}{let t=n-`
${o}`.length;return{autocompletePrefix:"",text:e.substring(0,t)+e.substring(n),selection:[t,t],commandId:s.delete,writeSelection:[null,null]}}}function _(e,t,n){let r=t[0]||0,l=t[1]||r;if(null===t[0]||r===l)return;let i=e.substring(0,r).lastIndexOf("\n")+1,o=e.indexOf("\n",l-1),a=o>0?o:e.length-1,c=e.substring(i,a).split("\n"),u=!1,d=0,f=0,m=[];for(let e of c){let t=e.match(/^\s*/);if(t){let s=t[0],r=e.substring(s.length);if(n){let e=s.length;s=s.slice(0,-2),d=u?d:s.length-e,u=!0,f+=s.length-e}else s+="  ",d=2,f+=2;m.push(s+r)}}let p=m.join("\n"),h=e.substring(0,i)+p+e.substring(a),b=[Math.max(i,r+d),l+f];return{text:h,selection:b,autocompletePrefix:p,commandId:s.insertText,writeSelection:[i,a]}}function j(e){let t=e.closest("task-lists");if(!t)throw Error("parent not found");let n=Array.from(t.querySelectorAll("ol, ul")).filter(e=>!e.closest("tracking-block"));return n.indexOf(e)}function C(e){let t=e.target;"backward"===t.selectionDirection?t.selectionEnd=t.selectionStart:t.selectionStart=t.selectionEnd}function R(e){if(0===document.querySelectorAll("tracked-issues-progress").length)return;let t=e.closest(".js-timeline-item");if(t)return;let n=e.querySelectorAll(".js-comment-body [type=checkbox]"),s=n.length,r=Array.from(n).filter(e=>e.checked).length,l=document.querySelectorAll("tracked-issues-progress[data-type=checklist]");for(let e of l)e.setAttribute("data-completed",String(r)),e.setAttribute("data-total",String(s))}function $(e){let t=e.closest(".contains-task-list"),n=t;for(;(n=n.parentElement.closest(".contains-task-list"))!==t&&null!==n;)t=n;return t}},19146:(e,t,n)=>{n.d(t,{W:()=>r});var s=n(59753);async function r(e){let t=document.querySelector("#site-details-dialog"),n=t.content.cloneNode(!0),r=n.querySelector("details"),l=r.querySelector("details-dialog"),i=r.querySelector(".js-details-dialog-spinner");e.detailsClass&&r.classList.add(...e.detailsClass.split(" ")),e.dialogClass&&l.classList.add(...e.dialogClass.split(" ")),e.label?l.setAttribute("aria-label",e.label):e.labelledBy&&l.setAttribute("aria-labelledby",e.labelledBy),document.body.append(n);try{let t=await e.content;i.remove(),l.prepend(t)}catch(n){i.remove();let t=document.createElement("span");t.textContent=e.errorMessage||"Couldn't load the content",t.classList.add("my-6"),t.classList.add("mx-4"),l.prepend(t)}return r.addEventListener("toggle",()=>{r.hasAttribute("open")||((0,s.f)(l,"dialog:remove"),r.remove())}),l}},34892:(e,t,n)=>{n.d(t,{DF:()=>m,Fr:()=>p,a_:()=>f});var s=n(67525);function r(e){let t=[...e.querySelectorAll("meta[name=html-safe-nonce]")].map(e=>e.content);if(t.length<1)throw Error("could not find html-safe-nonce on document");return t}let l=class ResponseError extends Error{constructor(e,t){super(`${e} for HTTP ${t.status}`),this.response=t}};function i(e,t,n=!1){let s=t.headers.get("content-type")||"";if(!n&&!s.startsWith("text/html"))throw new l(`expected response with text/html, but was ${s}`,t);if(n&&!(s.startsWith("text/html")||s.startsWith("application/json")))throw new l(`expected response with text/html or application/json, but was ${s}`,t);let r=t.headers.get("x-html-safe");if(r){if(!e.includes(r))throw new l("response X-HTML-Safe nonce did not match",t)}else throw new l("missing X-HTML-Safe nonce",t)}var o=n(71643),a=n(22490),c=n(46426),u=n(24601);let d=a.Z.createPolicy("server-x-safe-html",{createHTML:(e,t)=>{try{if((0,c.c)("BYPASS_TRUSTED_TYPES_POLICY_RULES"))return e;i(r(document),t)}catch(e){throw(0,u.eK)(e),(0,o.b)({incrementKey:"TRUSTED_TYPES_POLICY_ERROR"}),e}return e}});async function f(e,t,n){let r=new Request(t,n);r.headers.append("X-Requested-With","XMLHttpRequest");let l=await self.fetch(r);if(l.status<200||l.status>=300)throw Error(`HTTP ${l.status}${l.statusText||""}`);let i=d.createHTML(await l.text(),l);return(0,s.r)(e,i)}function m(e,t,n=1e3,s=[200]){return async function n(r){let l=new Request(e,t);l.headers.append("X-Requested-With","XMLHttpRequest");let i=await self.fetch(l);if(202===i.status)return await new Promise(e=>setTimeout(e,r)),n(1.5*r);if(s.includes(i.status))return i;if(i.status<200||i.status>=300)throw Error(`HTTP ${i.status}${i.statusText||""}`);throw Error(`Unexpected ${i.status} response status from poll endpoint`)}(n)}async function p(e,t,n){let{wait:s=500,acceptedStatusCodes:r=[200],max:l=3,attempt:i=0}=n||{},o=async()=>new Promise((n,o)=>{setTimeout(async()=>{try{let s=new Request(e,t);s.headers.append("X-Requested-With","XMLHttpRequest");let o=await self.fetch(s);if(r.includes(o.status)||i+1===l)return n(o);n("retry")}catch(e){o(e)}},s*i)}),a=await o();return"retry"!==a?a:p(e,t,{wait:s,acceptedStatusCodes:r,max:l,attempt:i+1})}},254:(e,t,n)=>{n.d(t,{ZG:()=>o,q6:()=>c,w4:()=>a});var s=n(8439);let r=!1,l=new s.Z;function i(e){let t=e.target;if(t instanceof HTMLElement&&t.nodeType!==Node.DOCUMENT_NODE)for(let e of l.matches(t))e.data.call(null,t)}function o(e,t){r||(r=!0,document.addEventListener("focus",i,!0)),l.add(e,t),document.activeElement instanceof HTMLElement&&document.activeElement.matches(e)&&t(document.activeElement)}function a(e,t,n){function s(t){let r=t.currentTarget;r&&(r.removeEventListener(e,n),r.removeEventListener("blur",s))}o(t,function(t){t.addEventListener(e,n),t.addEventListener("blur",s)})}function c(e,t){function n(e){let{currentTarget:s}=e;s&&(s.removeEventListener("input",t),s.removeEventListener("blur",n))}o(e,function(e){e.addEventListener("input",t),e.addEventListener("blur",n)})}},40458:(e,t,n)=>{n.d(t,{Z:()=>m});var s=n(19146),r=n(34892);function l(e){return new Promise(t=>{e.addEventListener("dialog:remove",t,{once:!0})})}function i(e){let t=document.querySelector(".sso-modal");t&&(t.classList.remove("success","error"),e?t.classList.add("success"):t.classList.add("error"))}function o(e){let t=document.querySelector("meta[name=sso-expires-around]");t&&t.setAttribute("content",e)}async function a(){let e=document.querySelector("link[rel=sso-modal]"),t=await (0,s.W)({content:(0,r.a_)(document,e.href),dialogClass:"sso-modal"}),n=null,a=window.external;if(a.ssoComplete=function(e){e.error?i(n=!1):(i(n=!0),o(e.expiresAround),window.focus()),a.ssoComplete=null},await l(t),!n)throw Error("sso prompt canceled")}function c(e){if(!(e instanceof HTMLMetaElement))return!0;let t=parseInt(e.content),n=new Date().getTime()/1e3;return n>t}async function u(){let e=document.querySelector("link[rel=sso-session]"),t=document.querySelector("meta[name=sso-expires-around]");if(!(e instanceof HTMLLinkElement)||!c(t))return!0;let n=e.href,s=await fetch(n,{headers:{Accept:"application/json","X-Requested-With":"XMLHttpRequest"}}),r=await s.json();return r}(0,n(36071).N7)(".js-sso-modal-complete",function(e){if(window.opener&&window.opener.external.ssoComplete){let t=e.getAttribute("data-error"),n=e.getAttribute("data-expires-around");window.opener.external.ssoComplete({error:t,expiresAround:n}),window.close()}else{let t=e.getAttribute("data-fallback-url");t&&(window.location.href=t)}});let d=null;function f(){d=null}async function m(){let e=await u();e||(d||(d=a().then(f).catch(f)),await d)}},46426:(e,t,n)=>{n.d(t,{$:()=>a,c:()=>i});var s=n(15205);let r=(0,s.Z)(l);function l(){return(document.head?.querySelector('meta[name="enabled-features"]')?.content||"").split(",")}let i=(0,s.Z)(o);function o(e){return -1!==r().indexOf(e)}let a={isFeatureEnabled:i}},56959:(e,t,n)=>{n.d(t,{RB:()=>s,qC:()=>r,w0:()=>Subscription});let Subscription=class Subscription{constructor(e){this.closed=!1,this.unsubscribe=()=>{e(),this.closed=!0}}};function s(e,t,n,s={capture:!1}){return e.addEventListener(t,n,s),new Subscription(()=>{e.removeEventListener(t,n,s)})}function r(...e){return new Subscription(()=>{for(let t of e)t.unsubscribe()})}}}]);
//# sourceMappingURL=app_assets_modules_github_behaviors_keyboard-shortcuts-helper_ts-app_assets_modules_github_be-f5afdb-bf7fe55bc98b.js.map