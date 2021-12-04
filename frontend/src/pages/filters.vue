<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" filterable v-model="params.model">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">Weights</div>
                <el-input class="value" size="small" v-model="params.pth" />
            </div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="layer" v-for="(layer, index) in res" :key="index">
                <div class="name">{{layer.name}}</div>
                <div class="filters">
                    <div class="filter" v-for="(filter, index) in layer.filters" :key="index">
                        <img class="pixelated" :src="filter" :title="index"/>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            models: [],
            images: {},
            res: [],
            params: {
                model: 'regnet_x_400mf',
                pth: null
            }
        };
    },
    created() {
        this.config()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        response_filters(data) {
            this.res = data
        },
        logs(data) {
            console.log(data)
        },
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
        },
        update() {
            this.$socket.emit("filters", this.params);
        },
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        background-color:white !important;
        display: flex;
        flex-flow: column;

        .layer {
            display: flex;
            flex-flow: row;
            align-items: center;
            margin: 3px 0;

            border-bottom: 1px dashed #bbb;

            .name {
                flex: 0 0 85px;
            }

            .filters {
                flex: 1 1 auto;
                display: flex;
                flex-flow: row;
                flex-wrap: wrap;

                .filter {
                    img {
                        width: 24px;
                        padding: 0 1px;
                        // height: 30px;
                    }
                }
            }
        }

        .layer:last-child {
            border: none;
        }
    }
}

</style>
